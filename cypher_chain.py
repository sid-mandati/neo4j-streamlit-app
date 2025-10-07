import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# NOTE: The manual schema has been removed to ensure compatibility.
# This will result in a slow startup time for the application.

cypher_examples = [
    {
        "question": "Which machine had the highest number of downtime events?",
        "query": """MATCH (m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
                    WITH m, COUNT(d) AS downtime_events
                    RETURN m.machine_description AS machine, downtime_events
                    ORDER BY downtime_events DESC
                    LIMIT 1;""",
    },
    {
        "question": "What is the total downtime in minutes for the 'Line 5 Filler / Capper'?",
        "query": """MATCH (m:Machine {machine_description: 'Line 5 Filler / Capper'})-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
                    RETURN SUM(d.downtime_in_minutes) AS total_downtime_minutes;""",
    },
    {
        "question": "What was the single most frequent fault description across all machines?",
        "query": """MATCH (f:MachineFault)<-[:DUE_TO_FAULT]-(d:MachineDowntimeEvent)
                    WITH f, COUNT(d) AS frequency
                    RETURN f.fault_description AS fault, frequency
                    ORDER BY frequency DESC
                    LIMIT
