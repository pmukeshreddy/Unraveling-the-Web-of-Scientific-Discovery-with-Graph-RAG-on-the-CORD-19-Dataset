# establish connection with neo4j

import time
import json
import re
from neo4j import  GraphDatabase
from tqdm.auto import tqdm


NEO4J_URI = "neo4j+s://your_aura_db_uri.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_aura_db_password"



driver = None
def connect_to_neo4j():
    global driver
    try:
        driver =  GraphDatabase.driver(NEO4J_URI,auth=(NEO4J_USER,NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j AuraDB.")
        return driver
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return None

def close_neo4j_connection():
    global driver
    if driver:
        driver.close()
        print("Neo4j connection closed.")
