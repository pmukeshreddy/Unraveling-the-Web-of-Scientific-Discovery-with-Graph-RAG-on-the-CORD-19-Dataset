import pandas as pd
from neo4j import GraphDatabase
import math

NEO4J_URL = "******"
NEO4J_USER = "*****"
NEO4J_PASSWORD = "******"

csv_path = "/kaggle/input/CORD-19-research-challenge/metadata.csv"

class PaperIngestor:
    def __init__(self,uri,user,password):
        self.driver = GraphDatabase.driver(uri,auth=(user,password))
    def close(self):
        self.driver.close()
    def creates_paper_nodes(self,csv_path):
        df = pd.read_csv(csv_path)
        df = df.where(pd.notnull(df),None)
        query = """
        MERGE (p:Paper {cord_uid: $cord_uid})
        SET p.title = $title,
            p.abstract = $abstract,
            p.publish_time = $publish_time,
            p.doi = $doi,
            p.pubmed_id = $pubmed_id,
            p.source_x = $source_x
        """
        with self.driver.session(database="neo4j") as session:
            for index , row in df.iterrows():
                session.run(query,**row.to_dict())
