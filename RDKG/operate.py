import ast
import asyncio
import re
import sys
from typing import Union
from collections import defaultdict


from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_single_entity_extraction(
    triple: list[str],
    chunk_key: str,
    attribute_list: list[str],
    context_base: dict,
    index: int
):
    if len(triple) < 3:
        return None
    # add this record as a node in the G
    entity_name = re.sub(r"^['\"]|['\"]$", "", clean_str(triple[index]).lower())
    entity_description=""
    for attribute in attribute_list:
        attribute = re.search(r"\((.*)\)", attribute)
        if attribute is None:
            continue
        attribute = attribute.group(1)
        entity_attribute = split_string_by_multi_markers(
            attribute, [context_base["tuple_delimiter"]]
        )

        entity_attribute_name = re.sub(r"^['\"]|['\"]$", "", clean_str(entity_attribute[0]).lower())
        if len(entity_attribute)>1:
            entity_description = clean_str(entity_attribute[1])
        else:
            continue

        if entity_name == entity_attribute_name:
            break

    if not entity_name.strip():
        return None
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_description=entity_description,
        source_id = entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3:
        return None
    # add this record as edge
    source = clean_str(record_attributes[0])
    target = clean_str(record_attributes[2])

    edge_id = clean_str(record_attributes[1])
    edge_source_id = chunk_key

    return dict(
        src_id=source,
        tgt_id=target,
        edge_id=edge_id,
        source_id=edge_source_id,
    )

async def _handle_llm_triple(
    llm1_result: list[list],
    llm2_result: list[list],
    llm3_result: list[list],
    content: str,
    ask_prompt: str,
    simple_prompt: str,
    different_prompt:str,
    context_base: dict,
    use_llm1_func: callable,
    use_llm2_func: callable,
    use_llm3_func: callable,
    use_llm4_func: callable,
):
    game_result = []
    # 提取相同项
    hint_prompt = simple_prompt.format(**context_base,llm1=str(llm1_result),llm2=str(llm2_result),llm3=str(llm3_result))
    simple_triple_result: str =await use_llm4_func(hint_prompt)
    same_result=process_to_list(simple_triple_result,context_base)
    game_result.extend(same_result)

    # 提取不同项
    hint_prompt = different_prompt.format(**context_base,llm1=str(llm1_result),llm2=str(llm2_result),llm3=str(llm3_result),simple_text=simple_prompt)
    different_triple_result: str =await use_llm4_func(hint_prompt)
    before_game_result=process_to_list(different_triple_result,context_base)

    # 询问llm1,llm2,llm3
    for result in before_game_result:
        hint_prompt = ask_prompt.format(**context_base, input_text=content, judge_text=result)
        if_agree_result1: str = await use_llm1_func(hint_prompt)
        if_agree_result1 = if_agree_result1.split('\n')[0].strip().strip('"').strip("'").lower()
        if if_agree_result1 == "yes":
            if_agree_result2: str = await use_llm2_func(hint_prompt)
            if_agree_result2 = if_agree_result2.split('\n')[0].strip().strip('"').strip("'").lower()
            if if_agree_result2 == "yes":
                if_agree_result3: str = await use_llm3_func(hint_prompt)
                if_agree_result3 = if_agree_result3.split('\n')[0].strip().strip('"').strip("'").lower()
                if if_agree_result3 == "yes":
                    game_result.append(result)
                else:
                    continue
            else:
                continue
        else:
            continue
    return game_result



async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
):
    already_source_ids = []
    already_description = []
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["entity_description"])

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    entity_description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["entity_description"] for dp in nodes_data] + already_description))
    )

    node_data = dict(
        source_id=source_id,
        entity_description=entity_description
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _insert_edges(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):

    edge_id = GRAPH_FIELD_SEP.join(
        set([dp["edge_id"] for dp in edges_data])
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data])
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            edge_id = edge_id,
            source_id = source_id
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        edge_id=edge_id
    )

    return edge_data


def contains_complete_sentence(text):
    # 匹配以标点符号（如 . ? !）结尾的句子
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return len(sentences) > 0  # 如果至少找到一个句子

def process_to_list(text,context_base):
    list_result=[]
    text_process1 = split_string_by_multi_markers(
        text,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    for record in text_process1:  # 按照标记分割出规整的三元组
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        list_result.append(record_attributes)
    return list_result

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    # 定义实体提取的模型和prompt
    use_llm1_func: callable = global_config["llm_model_func1"]
    use_llm2_func: callable = global_config["llm_model_func2"]
    use_llm3_func: callable = global_config["llm_model_func3"]
    use_llm4_func: callable = global_config["llm_model_func4"]
    embedding_func: callable = global_config["embedding_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    description_promopt = PROMPTS["generation_description"]
    single_entity_extract_prompt = PROMPTS["single_extraction"]
    overall_entity_extract_prompt = PROMPTS["triple_tuple_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    ask_prompt = PROMPTS["entity_agree_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]
    normalize_prompt = PROMPTS["normalize_process"]
    simple_prompt = PROMPTS["extraction_simple_triple"]
    different_prompt = PROMPTS["extraction_different_triple"]

    already_processed = 0
    already_entities = 0
    already_relations = 0
    # 处理单个
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        parts = content.split(".\\n\\n")
        extract_result1 = ""
        extract_result2 = ""
        extract_result3 = ""
        final_result = []
        all_attribute_list = []
        hint_prompt = ""
        # 分段提取
        for part in parts:
            if contains_complete_sentence(part):
                hint_prompt = single_entity_extract_prompt.format(**context_base, input_text=part)  # 将part加入模板中
                # print(hint_prompt)
                part_result1 = await use_llm1_func(hint_prompt)
                list_result1 = process_to_list(part_result1, context_base)  # 处理为列表格式，清除无关内容
                extract_result1 += part_result1

                part_result2 = await use_llm2_func(hint_prompt)
                list_result2 = process_to_list(part_result2, context_base)
                extract_result2 += part_result2

                part_result3 = await use_llm3_func(hint_prompt)
                list_result3 = process_to_list(part_result3, context_base)
                extract_result3 += part_result3

                # 三个模型进行博弈
                game_result = await _handle_llm_triple(
                    list_result1, list_result2, list_result3,
                    content, ask_prompt, simple_prompt, different_prompt, context_base,
                    use_llm1_func, use_llm2_func, use_llm3_func, use_llm4_func
                )
                # 规范化
                normalize_result=[]
                normalize_hint_prompt = normalize_prompt.format(input_text=game_result)
                normalize_result_str = await use_llm4_func(normalize_hint_prompt)
                try:
                    normalize_result = ast.literal_eval(normalize_result_str)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing normalize_result_str: {e}")
                    try:
                        normalize_result_str = await use_llm2_func(normalize_hint_prompt)
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing normalize_result_str: {e}")
                        normalize_result = []
                final_result.extend(normalize_result)

                # 提取实体与关系属性
                entity_list = []
                for i in range(len(normalize_result)):
                    if len(normalize_result[i]) == 3:
                        entity_list.append(normalize_result[i][0])
                        entity_list.append(normalize_result[i][2])
                print(entity_list)
                generation_description_prompt = description_promopt.format(**context_base, context=part, input_list=entity_list)
                result = await use_llm2_func(generation_description_prompt)
                attribute_list = split_string_by_multi_markers(
                    result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )
                all_attribute_list.extend(attribute_list)
                print(attribute_list)

            else:
                continue

        if hint_prompt:  # 确保 hint_prompt 已经被赋值
            history1 = pack_user_ass_to_openai_messages(hint_prompt, extract_result1)
            history2 = pack_user_ass_to_openai_messages(hint_prompt, extract_result2)
            history3 = pack_user_ass_to_openai_messages(hint_prompt, extract_result3)
        else:
            history1 = []

        # 整体提取和完整性检测
        for now_glean_index in range(entity_extract_max_gleaning):
            overall_hint_prompt = overall_entity_extract_prompt.format(**context_base, input_text=content)
            # 整体提取
            glean_result1 = await use_llm1_func(overall_hint_prompt)
            glean_list1 = process_to_list(glean_result1, context_base)
            glean_result2 = await use_llm2_func(overall_hint_prompt)
            glean_list2 = process_to_list(glean_result2, context_base)
            glean_result3 = await use_llm3_func(overall_hint_prompt)
            glean_list3 = process_to_list(glean_result3, context_base)
            # 博弈
            game_all_result = await _handle_llm_triple(
                glean_list1, glean_list2, glean_list3,
                content, ask_prompt, simple_prompt, different_prompt, context_base,
                use_llm1_func, use_llm2_func, use_llm3_func,use_llm4_func
            )

            # 规范化
            normalize_all_hint_prompt = normalize_prompt.format(input_text=game_all_result)
            normalize_result_str = await use_llm4_func(normalize_all_hint_prompt)
            try:
                normalize_result = ast.literal_eval(normalize_result_str)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing normalize_result_str: {e}")
                normalize_result_str = await use_llm2_func(normalize_all_hint_prompt)
                try:
                    normalize_result = ast.literal_eval(normalize_result_str)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing normalize_result_str: {e}")
                    normalize_result = []

            final_result.extend(normalize_result)

            # 提取属性
            entity_list = []
            for i in range(len(normalize_result)):
                if len(normalize_result[i]) == 3:
                    entity_list.append(normalize_result[i][0])
                    entity_list.append(normalize_result[i][2])
            print(entity_list)
            generation_description_prompt = description_promopt.format(**context_base, context=content,
                                                                       input_list=entity_list)
            result = await use_llm2_func(generation_description_prompt)
            attribute_list = split_string_by_multi_markers(
                result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            all_attribute_list.extend(attribute_list)
            print(attribute_list)

            # 历史数据记录，循环次数控制
            history1 += pack_user_ass_to_openai_messages(overall_hint_prompt, glean_result1)
            history2 += pack_user_ass_to_openai_messages(overall_hint_prompt, glean_result2)
            history3 += pack_user_ass_to_openai_messages(overall_hint_prompt, glean_result3)
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 判断是否提取完全
            if_loop_result1: str = await use_llm1_func(
                if_loop_prompt, history_messages=history1
            )
            if_loop_result1 = if_loop_result1.strip().strip('"').strip("'").lower()
            if if_loop_result1 != "yes":
                break

            if_loop_result2: str = await use_llm2_func(
                if_loop_prompt, history_messages=history2
            )
            if_loop_result2 = if_loop_result2.strip().strip('"').strip("'").lower()
            if if_loop_result2 != "yes":
                break

            if_loop_result3: str = await use_llm3_func(
                if_loop_prompt, history_messages=history3
            )
            if_loop_result3 = if_loop_result3.strip().strip('"').strip("'").lower()
            if if_loop_result3 != "yes":
                break
        print(final_result)
        # 存储
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for result in final_result:
            if_source_entities = await _handle_single_entity_extraction(
                result, chunk_key, all_attribute_list, context_base,0
            )
            if_target_entities = await _handle_single_entity_extraction(
                result, chunk_key, all_attribute_list, context_base,2
            )
            if if_source_entities is not None:
                maybe_nodes[if_source_entities["entity_name"]].append(if_source_entities)
            if if_target_entities is not None:
                maybe_nodes[if_target_entities["entity_name"]].append(if_target_entities)

            if_relation = await _handle_single_relationship_extraction(
                result, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities, {already_relations} relations\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst)  # 合并而不更新节点信息
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _insert_edges(k[0], k[1], v, knowledge_graph_inst, global_config) # 存储边信息
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"]+dp["source_id"],
                "entity_name": dp["entity_name"]
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["edge_id"]
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst
