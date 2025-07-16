from neo4j import GraphDatabase
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Embedder for distractor selection
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(df_chunks['chunk_text'].tolist())  # Reuse from previous


def run_cypher_query(query, params=None):
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]


def generate_raft_example(chunk_idx,num_distractors=3):
    oracle_chunk = df_chunks.iloc[chunk_idx]
    chunk_id = oracle_chunk['chunk_id']

    # Step 1: Generate question based on graph (e.g., related entities)
    entity_query = """
    MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
    RETURN e.name AS entity LIMIT 5
    """

    entities = run_cypher_query(entity_query, {'chunk_id': chunk_id})
    entities = [e['entity'] for e in entities] if entities else ['COVID-19']  # Fallback
    question = f"What is the relationship between {entities[0]} and related entities in the {oracle_chunk['section_title']} of {oracle_chunk['paper_title']}?"


    # Step 2: Oracle document: Chunk text + related graph context (entities/relations)
    graph_context_query = """
    MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (e)<-[:MENTIONS]-(other:Chunk)-[:MENTIONS]->(otherE:Entity)
    RETURN c.text AS chunk_text, collect(DISTINCT e.name) AS entities,
           collect(DISTINCT other.text) AS related_chunks LIMIT 3
    """

    graph_data = run_cypher_query(graph_context_query, {'chunk_id': chunk_id})
    if graph_data:
        oracle_doc = graph_data[0]['chunk_text'] + "\nRelated Entities: " + ", ".join(graph_data[0]['entities']) + \
                     "\nRelated Chunks: " + " | ".join(graph_data[0]['related_chunks'] or [])
    else:
        oracle_doc = oracle_chunk['chunk_text']

    # Step 3: Distractors: Low-sim chunks, but filter via graph (unrelated entities)
    similarities = cosine_similarity([chunk_embeddings[chunk_idx]], chunk_embeddings)[0]
    distractor_idxs = similarities.argsort()[:num_distractors + 10]  # Oversample
    distractor_docs = []
    for idx in distractor_idxs:
        dist_chunk_id = df_chunks.iloc[idx]['chunk_id']
        # Check if unrelated via graph
        relation_check = """
        MATCH (c:Chunk {id: $oracle_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(d:Chunk {id: $dist_id})
        RETURN count(e) AS common
        """
        common = run_cypher_query(relation_check, {'oracle_id': chunk_id, 'dist_id': dist_chunk_id})
        if common and common[0]['common'] == 0:  # No shared entities
            distractor_docs.append(df_chunks.iloc[idx]['chunk_text'])
        if len(distractor_docs) == num_distractors:
            break


    
    cot_answer = f"From the graph-enriched chunk: {oracle_doc[:200]}... The relation involves [entities], leading to [reasoned summary]. Ignore unrelated: {distractor_docs[0][:50]}..."
    
    # Randomly remove oracle for memorization training (20%)
    if random.random() < 0.2:
        docs = distractor_docs
    else:
        docs = [oracle_doc] + distractor_docs
        random.shuffle(docs)
    
    return {
        'question': question,
        'documents': docs,'answer': cot_answer
    }
