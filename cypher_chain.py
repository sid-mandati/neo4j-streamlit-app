import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from schema_builder import build_enriched_schema

load_dotenv()

# Build the schema automatically on startup
graph_schema = build_enriched_schema()

# Define Few-Shot Examples (with corrected traversal for maintenance correlation)
cypher_examples = [
    {
        "question": "Which machine had the most downtime events?",
        "query": """MATCH (m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
                    WITH m, COUNT(d) AS downtime_events
                    RETURN m.machine_description AS machine, downtime_events
                    ORDER BY downtime_events DESC
                    LIMIT 1;""",
    },
    {
