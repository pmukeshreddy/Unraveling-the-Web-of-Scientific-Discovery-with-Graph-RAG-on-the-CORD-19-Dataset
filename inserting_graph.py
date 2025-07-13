#one function creates right prompt for llm and one to call llmm and get data

def get_extraction_prompt(text_chunk):
    return f"""
    You are an expert biomedical researcher. Your task is to extract entities and relationships from the provided text based on the schema.
    Schema:
    - Nodes: Disease, Virus, Gene, Protein, Drug, Treatment, Symptom
    - Edges: TREATS, CAUSES, INHIBITS, EXPRESSES, ASSOCIATED_WITH
    Instructions: Return a single, valid JSON object. If nothing is found, return an empty JSON object: {{\"entities\": [], \"relationships\": []}}.
    Text to Analyze: "{text_chunk}"
    Output (JSON format):
    """

def extract_entities_and_relationships(response_text):
    if "Error:" in response_text:
        # print(f"API call failed: {response_text}") # Uncomment for deeper debugging
        return {}
    try:
        json_match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", response_text, re.DOTALL)
        clean_json = json_match.group(1) if json_match else response_text
        return json.loads(clean_json)
    except json.JSONDecodeError:
        # print(f"Could not parse JSON from response: {response_text}") # Uncomment for deeper debugging
        return {}


def add_batch_to_graph(tx, batch):
    cypher_query = """
    UNWIND $batch AS row
    MERGE (p:Paper {paper_id: row.paper_id}) ON CREATE SET p.title = row.paper_title
    MERGE (c:Chunk {chunk_id: row.chunk_id}) ON CREATE SET c.text = row.chunk_text
    MERGE (p)-[:HAS_CHUNK]->(c)
    WITH c, row
    UNWIND row.entities AS entity
    MERGE (e:Entity {name: entity.id}) ON CREATE SET e.type = entity.type
    MERGE (c)-[:MENTIONS]->(e)
    WITH c, row
    UNWIND row.relationships AS rel
    MATCH (source:Entity {name: rel.source})
    MATCH (target:Entity {name: rel.target})
    CALL apoc.create.relationship(source, rel.type, {{}}, target) YIELD rel
    RETURN count(*)
    """
    tx.run(cypher_query, batch=batch)    

async def process_and_populate_graph(df_chunks_to_process, api_batch_size=50):
    global driver, GEMINI_API_KEY, HEADERS
    
    # This semaphore is the rate-limiter. It allows 2 calls at once.
    semaphore = asyncio.Semaphore(2)
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    async with aiohttp.ClientSession(headers=HEADERS) as http_session:
        db_batch = []
        for i in tqdm(range(0, len(df_chunks_to_process), api_batch_size), desc="Processing API Batches"):
            chunk_batch = df_chunks_to_process.iloc[i:i + api_batch_size]
            
            tasks = []
            for row in chunk_batch.itertuples(index=False):
                prompt = get_extraction_prompt(row.chunk_text)
                async def task_with_semaphore(p, r):
                    async with semaphore:
                        return await async_call_gemini_api(http_session, p, api_url), r
                tasks.append(task_with_semaphore(prompt, row))

            api_results = await asyncio.gather(*tasks)

            for response_text, row in api_results:
                extracted_data = extract_entities_and_relationships(response_text)
                if extracted_data:
                    db_batch.append({
                        "paper_id": row.paper_id, "paper_title": row.paper_title,
                        "chunk_id": row.chunk_id, "chunk_text": row.chunk_text,
                        "entities": extracted_data.get('entities', []),
                        "relationships": extracted_data.get('relationships', [])
                    })
            
            # Write the accumulated results to Neo4j
            if db_batch:
                with driver.session(database="neo4j") as db_session:
                    db_session.execute_write(add_batch_to_graph, db_batch)
                print(f"  > Wrote {len(db_batch)} extracted results to Neo4j.")
                db_batch = []
