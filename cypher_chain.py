import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the Schema ---
# This remains the same.
graph_schema = """
# Node Labels and Properties
(:MaintenanceWorkOrder {work_order_id: 'INTEGER', work_order_description: 'STRING', maintenance_type: 'STRING'})
(:Equipment {sap_equipment_number: 'STRING', sap_equipment_description: 'STRING'})
(:MachineDowntimeEvent {event_start_datetime: 'DATETIME', downtime_in_minutes: 'FLOAT'})
(:Machine {machine_id: 'STRING', machine_description: 'STRING', plant_id: 'INTEGER', location_id: 'STRING'})
(:Location {location_id: 'STRING', location_name: 'STRING', plant_id: 'INTEGER'})
(:MachineFault {plant_line_machine_fault_code_id: 'STRING', fault_description: 'STRING', fault_category: 'STRING'})

# Relationships
(:Machine)-[:FALLS_UNDER]->(:Location)
(:Machine)-[:PROCESS_FLOWS_TO]->(:Machine)
(:Machine)-[:CAN_FAULT_DUE_TO]->(:MachineFault)
(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)
(:Equipment)-[:MAPS_TO]->(:Machine)
(:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(:MachineFault)
(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)
(:MaintenanceWorkOrder)-[:PERFORMED_AT_LOCATION]->(:Location)
"""

# --- 2. Create Few-Shot Examples ---
# These examples are still very useful for pattern recognition.
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
                    LIMIT 1;""",
    },
    {
        "question": "Show me 5 maintenance work orders.",
        "query": """MATCH (wo:MaintenanceWorkOrder)
                    RETURN wo.work_order_id, wo.work_order_description
                    LIMIT 5;""",
    },
]

# --- 3. Create the NEW, Stricter Custom Prompt Template ---
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query developer.
Your ONLY task is to write a single, syntactically correct Cypher query to answer the user's question.
DO NOT add any text before or after the query. DO NOT explain the query.

You must follow these strict rules:
1.  **Use ONLY the nodes, relationships, and properties provided in the Schema.** Do not hallucinate or invent any that are not listed.
2.  **Follow the graph structure.** Do not create paths that do not exist in the schema. For example, the path `(:Location)-[:DUE_TO_FAULT]->(:MachineFault)` is INVALID because Locations do not have a DUE_TO_FAULT relationship. The correct way to find faults for a machine in a location is `(:Location)<-[:FALLS_UNDER]-(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(:MachineFault)`.
3.  **Identify entities correctly.** When a question mentions a specific machine by name, like 'Line 5 Filler / Capper', you MUST filter on the `machine_description` property of the `Machine` node. DO NOT mistake it for a Location.
4.  **Count events, not nodes.** When asked for the "frequency" or "number of times" a fault occurs, you MUST count the `MachineDowntimeEvent` nodes associated with that fault. DO NOT simply count the `MachineFault` nodes.
5.  **Always return properties.** Do not return entire nodes. For example, instead of `RETURN m`, use `RETURN m.machine_description`.

Schema:
{schema}

---
Here are some examples of questions and their correct Cypher queries. Use them to learn the correct patterns.
{examples}
---

The question is:
{question}
"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "examples"], template=CYPHER_GENERATION_TEMPLATE
)

# --- 4. The Connector Class ---
# This remains the same, but will now use the new, stricter prompt.
class Neo4jLLMConnector:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            schema=graph_schema
        )
        self.llm = ChatOpenAI(temperature=0)
        
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            cypher_prompt=CYPHER_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

    def ask(self, question):
        try:
            result = self.chain.invoke({"query": question, "examples": cypher_examples})
            
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "Query not generated.")
            final_answer = result.get("result", "Could not find an answer.")
            
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred", str(e)

