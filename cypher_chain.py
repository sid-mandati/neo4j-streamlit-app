import os
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- START: Schema Builder Logic ---
# This logic is now self-contained in this file.

def get_distinct_values(session, node_label, property_name):
    """Helper function to run a DISTINCT query."""
    query = f"MATCH (n:{node_label}) WHERE n.{property_name} IS NOT NULL RETURN DISTINCT n.{property_name} AS values"
    result = session.run(query)
    return [record["values"] for record in result]

def build_enriched_schema():
    """
    Connects to Neo4j, fetches distinct values, and returns a formatted schema string.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        order_status_values = get_distinct_values(session, "MaintenanceWorkOrder", "order_status")
        maintenance_type_values = get_distinct_values(session, "MaintenanceWorkOrder", "maintenance_type")
        fault_category_values = get_distinct_values(session, "MachineFault", "fault_category")
    driver.close()

    schema = f"""
# Node Labels and Properties
(:MaintenanceWorkOrder {{work_order_id: 'INTEGER', maintenance_type: 'STRING' /* one of: {maintenance_type_values} */, order_status: 'STRING' /* one of: {order_status_values} */}})
(:Equipment {{sap_equipment_number: 'STRING', sap_equipment_description: 'STRING'}})
(:MachineDowntimeEvent {{downtime_in_minutes: 'FLOAT'}})
(:Machine {{machine_description: 'STRING'}})
(:Location {{location_name: 'STRING'}})
(:MachineFault {{fault_description: 'STRING', fault_category: 'STRING' /* one of: {fault_category_values} */}})

# Relationships
(:Machine)-[:FALLS_UNDER]->(:Location)
(:Machine)-[:PROCESS_FLOWS_TO]->(:Machine)
(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)
(:Equipment)-[:MAPS_TO]->(:Machine)
(:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(:MachineFault)
(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)
"""
    return schema

# --- END: Schema Builder Logic ---


# Build the schema automatically on startup
graph_schema = build_enriched_schema()

# Define Few-Shot Examples
cypher_examples = [
    {"question": "How many machines are there?", "query": "MATCH (m:Machine) RETURN count(m);"},
    {"question": "Which machine had the most downtime?", "query": "MATCH (m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d) RETURN m.machine_description, count(d) AS events ORDER BY events DESC LIMIT 1;"},
    {"question": "Find overdue maintenance work orders.", "query": "MATCH (wo:MaintenanceWorkOrder) WHERE wo.order_status IN ['In Progress', 'Not Started'] AND wo.planned_date < date() RETURN wo.work_order_description;"},
]

# Define the Custom Prompt Template
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j developer. Write a Cypher query to answer the user's question.

You must follow these strict rules:
1. Use ONLY the nodes, relationships, and properties in the provided Schema.
2. Use the provided values in comments when filtering (e.g., `/* one of: ... */`).
3. To find the "frequency" of a fault, count `MachineDowntimeEvent` nodes.
4. To correlate maintenance with downtime, use the path: `(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)-[:MAPS_TO]->(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)`.

Schema:
{schema}
---
Examples:
{examples}
---
Question: {question}"""

CYPHER_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# The Connector Class
class Neo4jLLMConnector:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        self.graph.schema = graph_schema
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            cypher_prompt=CYPHER_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            return_direct=True
        )

    def ask(self, question):
        try:
            result = self.chain.invoke({"query": question, "examples": str(cypher_examples)})
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "Query not generated.")
            final_answer = result.get("result", "Could not find an answer.")
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred", str(e)
