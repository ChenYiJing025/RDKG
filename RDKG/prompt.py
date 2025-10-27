GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["single_extraction_1"] = """
Input_text:{input_text}
Extract triples from this paragraph and output them in lowercase format. Use '{record_delimiter}' to separate each triple. When all triples are extracted, output <|COMPLETE|> with no additional text.
Output format:(<source>{tuple_delimiter}<relation>{tuple_delimiter}<target>){record_delimiter}
Example:
Text:You can usually get information on your local association from the county extension office or from the beekeeping equipment dealer in your area.
Output:
(you{tuple_delimiter}can get information on{tuple_delimiter}your local association){record_delimiter}
(information{tuple_delimiter}is available from{tuple_delimiter}the county extension office){record_delimiter}
(information{tuple_delimiter}is available from{tuple_delimiter}the beekeeping equipment dealer in your area){completion_delimiter}
"""

PROMPTS["single_extraction"] ="""
Input_text:{input_text}
Extract concise triples from this paragraph, focusing on noun-based phrases and minimizing modifiers. Use '{record_delimiter}' to separate each triple. When all triples are extracted, output <|COMPLETE|> with no additional text.

Output format: (<source>{tuple_delimiter}<relation>{tuple_delimiter}<target>){record_delimiter}

Example:
Text: You can usually get information on your local association from the county extension office or from the beekeeping equipment dealer in your area.

Output:
(you{tuple_delimiter}get{tuple_delimiter}information){record_delimiter}
(information{tuple_delimiter}from{tuple_delimiter}county extension office){record_delimiter}
(information{tuple_delimiter}from{tuple_delimiter}beekeeping equipment dealer){completion_delimiter}
"""

PROMPTS["triple_tuple_extraction"] = """
Task: Extract concise triples from this paragraph, focusing on noun-based phrases and minimizing modifiers. Use '{record_delimiter}' to separate each triple. When all triples are extracted, output <|COMPLETE|> with no additional text.
Output format:(<source>{tuple_delimiter}<relation>{tuple_delimiter}<target>){record_delimiter}
Example:
Text:You can usually get information on your local association from the county extension office or from the beekeeping equipment dealer in your area.
Output:
(you{tuple_delimiter}get{tuple_delimiter}information){record_delimiter}
(information{tuple_delimiter}from{tuple_delimiter}county extension office){record_delimiter}
(information{tuple_delimiter}from{tuple_delimiter}beekeeping equipment dealer){completion_delimiter}
Text: {input_text}
"""
PROMPTS["entity_agree_extraction"] = """
Evaluate if the given triple (subject, predicate, object) can be derived from the text. Answer YES|NO.
Input:
Text: {input_text}
Triple: {judge_text}
"""
PROMPTS["entity_extraction_zero_shot"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["extraction_simple_triple_no"] ="""
Task: Please help me extract the triples with the same meaning contained in all three lists and output them strictly in the specified format.
List1:{llm1}
List2:{llm2}
List3:{llm3}
Output format example is as follows:
(you{tuple_delimiter}can get information on{tuple_delimiter}your local association){record_delimiter}
(information{tuple_delimiter}is available from{tuple_delimiter}the county extension office){record_delimiter}
(information{tuple_delimiter}is available from{tuple_delimiter}the beekeeping equipment dealer in your area){completion_delimiter}
"""

PROMPTS["extraction_simple_triple"] ="""
Task: Extract the triples that have the same meaning and appear in all three lists. The output must **strictly** follow the specified format.

List1: {llm1}  
List2: {llm2}  
List3: {llm3}  

### **Output format (must follow strictly):**  
Each triple must be formatted as:  
(subject{tuple_delimiter}predicate{tuple_delimiter}object){record_delimiter}  
Ensure that the last triple ends with {completion_delimiter}.  

### **Example output (follow this format exactly):**  
(you{tuple_delimiter}can get information on{tuple_delimiter}your local association){record_delimiter}  
(information{tuple_delimiter}is available from{tuple_delimiter}the county extension office){record_delimiter}  
(information{tuple_delimiter}is available from{tuple_delimiter}the beekeeping equipment dealer in your area){completion_delimiter}  

### **Important rules (read carefully):**  
1. **Only extract triples that appear in all three lists and have the same meaning.**  
2. **Output must strictly match the format. Do NOT add extra text, explanations, or titles.**  
3. **Do NOT include triples that are different or appear in only one or two lists.**  
4. **Do NOT modify the structure of the triples. Keep them as they appear in the lists.**  
5. **Ensure the last triple ends with {completion_delimiter}.**  

⚠️ **Failure to follow the format exactly will result in incorrect output.**
"""

PROMPTS["extraction_different_triple"] ="""
Task: Extract the unique triples that appear in List1, List2, and List3 but are NOT in the "List of identical triples". Output only the different triples in the specified format.

List of identical triples: {simple_text}  
List1: {llm1}  
List2: {llm2}  
List3: {llm3}  

### Output Format:
Each triple should be formatted as:  
(subject{tuple_delimiter}predicate{tuple_delimiter}object){record_delimiter}  
Ensure that the last triple ends with {completion_delimiter}.  

#### **Example Output Format (DO NOT COPY THESE TRIPLES, JUST FOLLOW THE STRUCTURE)**:
(subject1{tuple_delimiter}predicate1{tuple_delimiter}object1){record_delimiter}  
(subject2{tuple_delimiter}predicate2{tuple_delimiter}object2){record_delimiter}  
(subject3{tuple_delimiter}predicate3{tuple_delimiter}object3){completion_delimiter}  

### **STRICT RULES (FOLLOW EXACTLY)**
- **ONLY extract unique triples from List1, List2, and List3 that are NOT in the "List of identical triples".**
- **DO NOT** include any triples from the example—those are just for formatting reference.
- **DO NOT** modify the extracted triples—keep them exactly as they appear in List1, List2, and List3.
- **DO NOT** output anything other than the required triples in the specified format.
- If there are no unique triples, return an **empty output** (do not add explanations).

⚠️ Any deviation from these instructions will be incorrect. Follow the format strictly.
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY triples were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some triples may have still been missed.  Answer YES | NO if there are still triples that need to be added.
"""

PROMPTS[
    "normalize_process"
] = """
Please normalize the following list of triples (subject, predicate, object) while adhering to these guidelines:
Lexical Standardization: Standardize the spelling and format of subjects and objects, removing unnecessary articles and ensuring singular/plural consistency.
Redundancy Removal: Eliminate redundant words, phrases, or unnecessary details that do not add meaningful information.
Consistent Format: Ensure all triples remain in the (subject, predicate, object) structure after normalization.
Output the result directly in the input format without any additional text.
Example Input:
[["honey", "became shipped by", "the carload"],["the bees", "are collected from", "flowers by the bees"]]
Example Output:
[["honey", "was shipped in", "carloads"],["bees", "collect nectar from", "flowers"]]
Now, please normalize the following triples accordingly:{input_text}
"""

PROMPTS[
    "generation_description"
]="""
Based on the text from these entity sources, help me write a description of each entity in the list and output it strictly  in the specified format. Below is the text:
{context}
###Here is the triple list:
{input_list}

###Output Format (Strictly Follow This Format):
Each entity description must be formatted as:
("Entity Name"{tuple_delimiter}"Entity Description"){record_delimiter}
Ensure that the last entity ends with {completion_delimiter} instead of {record_delimiter}.

###Example Output:
("Alex"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("Taylor"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){completion_delimiter}
"""
