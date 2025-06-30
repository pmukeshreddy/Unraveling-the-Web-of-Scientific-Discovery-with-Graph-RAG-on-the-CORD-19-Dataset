#one function creates right prompt for llm and one to call llmm and get data

def get_extraction_prompt(text_chunk):
    return f"""
    You are an expert biomedical researcher building a knowledge graph.
    Your task is to analyze the provided text and extract entities and relationships according to the specified schema.

    # Schema:
    - Nodes: Disease, Virus, Gene, Protein, Drug, Treatment, Symptom
    - Edges: TREATS, CAUSES, INHIBITS, EXPRESSES, ASSOCIATED_WITH

    # Instructions:
    - Only extract entities that match the node types in the schema.
    - Only extract relationships that match the edge types in the schema.
    - Return a single, valid JSON object with 'entities' and 'relationships' keys.
    - If no relevant entities or relationships are found, return an empty JSON object: {{"entities": [], "relationships": []}}.

    # Example:
    Text to Analyze: "The drug Remdesivir has been shown to inhibit the RNA polymerase protein of the SARS-CoV-2 virus, which causes COVID-19."
    Output (JSON format):
    {{
        "entities": [
            {{"id": "Remdesivir", "type": "Drug"}},
            {{"id": "RNA polymerase", "type": "Protein"}},
            {{"id": "SARS-CoV-2", "type": "Virus"}},
            {{"id": "COVID-19", "type": "Disease"}}
        ],
        "relationships": [
            {{"source": "Remdesivir", "target": "RNA polymerase", "type": "INHIBITS"}},
            {{"source": "SARS-CoV-2", "target": "COVID-19", "type": "CAUSES"}}
        ]
    }}

    # Your Task:
    Text to Analyze: "{text_chunk}"
    Output (JSON format):
    """


def extract_entities_and_realationships(text_chunk):
    if not isinstance(text_chunk,str) or len(text_chunk) < 5:
        return {}
    prompt = get_extraction_prompt(text_chunk)
    response_text = call_gemini_api(prompt)
    try:
        json_match = re.search(r"```json\\n(.*)```",response_text,re.DOTALL)
        clean_json = json_match.group(1) if json_match else response_text
        return json.loads(clean_json)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Could not parse JSON from response: {response_text}")
        return {}

def add_data_to_graph(tx,paper_id,paper_title,chunk_id,chunk_text,extracted_data):
    tx.run("MERGE (p:Paper {paper_id: $paper_id}) ON CREATE SET p.title = $paper_title",
           paper_id=paper_id, paper_title=paper_title) # upsert
    tx.run("MATCH (p:Paper {paper_id: $paper_id}) "
           "MERGE (c:Chunk {chunk_id: $chunk_id}) ON CREATE SET c.text = $chunk_text "
           "MERGE (p)-[:HAS_CHUNK]->(c)",
           paper_id=paper_id, chunk_id=chunk_id, chunk_text=chunk_text) #  it finds the existing paper , find or create text chunk , then ensure relationship exist for sure
     for entity in extracted_data.get('entities', []):
        tx.run(f"MERGE (e:{entity['label']} {{name: $name}}) "
               "WITH e "
               "MATCH (c:Chunk {{chunk_id: $chunk_id}}) "
               "MERGE (c)-[:MENTIONS]->(e)",
               name=entity['name'], chunk_id=chunk_id) # for each entity first it make sure it exist as node next it connects with chunk
    for rel in extracted_data.get('relationships', []):
        tx.run(f"MATCH (source {{name: $source_name}}) "
               f"MATCH (target {{name: $target_name}}) "
               f"MERGE (source)-[:{rel['type']}]->(target)",
               source_name=rel['source'], target_name=rel['target']) # it finds two node and make sure they have relationship
    

def process_and_populate_graph(df_chunks_to_process):
    """Main function to iterate through chunks, extract data, and populate the graph."""
    global driver
    if not driver or driver.is_closed():
        print("Cannot proceed. Neo4j connection is not available. Please run Cell 1.")
        return

    print(f"Starting to process {len(df_chunks_to_process)} chunks for graph population...")
    with driver.session(database="neo4j") as session:
        for index, row in tqdm(df_chunks_to_process.iterrows(), total=df_chunks_to_process.shape[0], desc="Populating Graph"):
            extracted_data = extract_entities_and_relationships(row['chunk_text'])
            if extracted_data and (extracted_data.get('entities') or extracted_data.get('relationships')):
                try:
                    session.execute_write(
                        add_data_to_graph,
                        paper_id=row['paper_id'], paper_title=row['paper_title'],
                        chunk_id=row['chunk_id'], chunk_text=row['chunk_text'],
                        extracted_data=extracted_data
                    )
                except Exception as e:
                    print(f"Error writing to graph for chunk {row['chunk_id']}: {e}")
            time.sleep(1) # Rate limiting to be kind to the API
