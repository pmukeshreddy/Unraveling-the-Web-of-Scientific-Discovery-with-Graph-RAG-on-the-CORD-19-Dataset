import random
from neo4j import GraphDatabase
import pandas as pd

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(query, params=None):
    with driver.session() as session:
        return [r.data() for r in session.run(query, params)]

# Pull a larger pool of entities for diversity
entities = [e['entity'] for e in run_query("MATCH (e:Entity) RETURN e.text AS entity LIMIT 200")]

# Multiple question templates for variation
QUESTION_TEMPLATES = [
    "Which papers mention {ent} and its top co‑occurring entities?",
    "Find publications that discuss {ent} along with related topics.",
    "List papers that mention {ent} together with other entities."
]

raft_data = []
for _ in range(1000):
    ent = random.choice(entities)
    question = random.choice(QUESTION_TEMPLATES).format(ent=ent)

    # Retrieve up to 3 papers and the top 10 distinct co-occurring entities
    oracle_query = """
      MATCH (e:Entity {text: $ent})<-[:MENTIONS]-(p:Paper)
      OPTIONAL MATCH (e)-[:CO_OCCURS]-(other:Entity)
      WITH p.paper_id AS paper, collect(DISTINCT other.text)[0..10] AS co_occurs
      RETURN paper, co_occurs
      LIMIT 3
    """
    oracle_rows = run_query(oracle_query, {'ent': ent})
    # Summarise the evidence instead of dumping a huge list
    oracle_docs = [
        f"Paper {row['paper']} mentions {ent} with related terms: {', '.join(row['co_occurs'])}"
        for row in oracle_rows
    ]

    # Sample three distractor entities that are not the target entity
    distractor_query = """
      MATCH (d:Entity) WHERE d.text <> $ent
      RETURN d.text AS dist_ent
      ORDER BY rand()
      LIMIT 3
    """
    dists = run_query(distractor_query, {'ent': ent})
    distractors = [f"Unrelated: {d['dist_ent']}" for d in dists]

    # Compose a short answer without chain-of-thought leakage
    if oracle_docs:
        answer = f"The relevant papers are: {', '.join(row['paper'] for row in oracle_rows)}."
    else:
        answer = "No papers were found that match the criteria."

    raft_data.append({
        'question': question,
        'documents': oracle_docs + distractors,
        'answer': answer
    })

pd.DataFrame(raft_data).to_json('raft_dataset.json', orient='records')
driver.close()
