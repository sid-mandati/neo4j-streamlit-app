import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            print(f"Failed to create Neo4j driver: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        if not self.driver:
            print("Neo4j driver not initialized.")
            return []
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

# Initialize a single connection object for the application to use
db_conn = Neo4jConnection(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)
