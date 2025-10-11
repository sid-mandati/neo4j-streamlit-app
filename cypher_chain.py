import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# The formatted schema from your database, for fast startup
graph_schema = """
# Node Labels and Properties
(:MaintenanceWorkOrder {work_order_id: 'INTEGER', maintenance_type: 'STRING' /* one of: ['Preventative Maintenance', 'Corrective Maintenance', 'Breakdown Maintenance'] */, order_status: 'STRING' /* one of: ['Completed OnTime', 'Completed Late', 'Not Started', 'Skipped', 'In Progress', 'Incomplete'] */, actual_finish_date: 'DATE'})
(:Equipment {sap_equipment_number: 'STRING', sap_equipment_description: 'STRING'})
(:MachineDowntimeEvent {event_start_datetime: 'DATETIME', downtime_in_minutes: 'FLOAT'})
(:Machine {machine_description: 'STRING'})
(:Location {location_name: 'STRING'})
(:MachineFault {fault_description: 'STRING', fault_category: 'STRING'})

# Relationships
(:Machine)-[:FALLS_UNDER]->(:Location)
(:Machine)-[:PROCESS_FLOWS_TO]->(:Machine)
(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)
(:Equipment)-[:MAPS_TO]->(:Machine)
(:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(:MachineFault)
(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)
(:MachineDowntimeEvent)-[:PRECEDES]->(:MachineDowntimeEvent)
"""

cypher_examples = [
    {"question": "How many machines are there?", "query": "MATCH (m:Machine) RETURN count(m);"},
    {"question": "Did any downtime occur on a machine within 7 days after maintenance?", "query": "MATCH (wo:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(e:Equipment)-[:MAPS_TO]->(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent) WHERE d.event_start_datetime > datetime(wo.actual_finish_date) AND d.event_start_datetime < datetime(wo.actual_finish_date) + duration({days: 7}) RETURN m.machine_description, wo.work_order_description LIMIT 5;"},
    {"question": "Find the root cause of a cascading failure.", "query": "MATCH path = (root_cause:MachineDowntimeEvent)-[:PRECEDES*]->(downstream_event:MachineDowntimeEvent) WHERE NOT (root_cause)<-[:PRECEDES]-() WITH path, length(path) AS len ORDER BY len DESC LIMIT 1 UNWIND nodes(path) AS event MATCH (event)<-[:RECORDED_DOWNTIME_EVENT]-(m:Machine) RETURN m.machine_description AS machine, event.event_start_datetime AS time ORDER BY time;"},
]

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j developer. Write a Cypher query to answer the user's question.

You must follow these strict rules:
1. Use ONLY the nodes, relationships, and properties in the provided Schema.
2. To find the root cause of a failure, traverse backwards using the `[:PRECEDES]` relationship.
3. Handle DATE to DATETIME conversion using the `datetime()` function before adding a duration.
4. Use the values in comments when filtering (e.g., `/* one of: ... */`).

Schema:
{schema}
---
Examples:
{examples}
---
Question: {question}"""

CYPHER_PROMPT = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

class Neo4jLLMConnector:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            schema=graph_schema  # Re-enabling for fast startup
        )
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
