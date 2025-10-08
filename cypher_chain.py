import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the high-quality, manual schema ---
graph_schema = """
# Node Labels and Properties
(:MaintenanceWorkOrder {work_order_id: 'INTEGER', work_order_description: 'STRING', maintenance_type: 'STRING', actual_finish_date: 'DATE'})
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

# --- 2. Define Few-Shot Examples ---
cypher_examples = [
    {
        "question": "How many machines are there in total?",
        "query": "MATCH (m:Machine) RETURN count(m) AS total_machines;"
    },
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
        "question": "What maintenance was done on the 'Line 7 Labeler'?",
        "query": """MATCH (m:Machine {machine_description: 'Line 7 Labeler'})<-[:MAPS_TO]-(e:Equipment)<-[:PERFORMED_ON_EQUIPMENT]-(wo:MaintenanceWorkOrder)
                    RETURN wo.work_order_description, wo.maintenance_type
                    LIMIT 5;""",
    },
]

# --- 3. Create the Custom Prompt Template ---
# This template now correctly expects 'schema' and 'question' as inputs.
# The examples are "baked in" as static text.

# First, format the examples into a plain string.
formatted_examples = "\n\n".join(
    [f"Question: {e['question']}\nQuery: ```cypher\n{e['query']}\n```" for e in cypher_examples]
)

# Next, create the template string, injecting the examples directly.
CYPHER_GENERATION_TEMPLATE = f"""You are an expert Neo4j Cypher query developer.
Your ONLY task is to write a single, syntactically correct Cypher query to answer the user's question.
DO NOT add any text before or after the query. DO NOT explain the query.

You must follow these strict rules:
1.  **Use ONLY the nodes, relationships, and properties provided in the Schema.**
2.  **Follow the graph structure.** Do not create paths that do not exist.
3.  **Identify entities correctly.** Filter on the correct properties for machines or locations.
4.  **Count events, not nodes.** To find the "frequency" of a fault, count `MachineDowntimeEvent` nodes.
5.  **Handle dates correctly.** Use `date()` and `duration()` for time-based questions.
6.  **Always return properties.** Do not return entire nodes.

Schema:
{{schema}}

---
Here are some examples of questions and their correct Cypher queries. Use them to learn the patterns.
{formatted_examples}
---

The question is:
{{question}}
"""

# The input variables now correctly match what the chain provides.
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# --- 4. The Connector Class ---
class Neo4jLLMConnector:
    def __init__(self):
        # Initialize the graph without the schema argument to avoid the TypeError
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
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
            # The invoke call is now simpler, as the chain handles all prompt variables.
            result = self.chain.invoke({"query": question})
            
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "Query not generated.")
            final_answer = result.get("result", "Could not find an answer.")
            
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred", str(e)

