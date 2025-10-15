import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from db_connector import db_conn

load_dotenv()

# --- START: Self-Contained Schema Builder Logic ---
def build_enriched_schema():
    def get_distinct_values(node_label, property_name):
        query = f"MATCH (n:{node_label}) WHERE n.{property_name} IS NOT NULL RETURN DISTINCT n.{property_name} AS values"
        results = db_conn.run_query(query)
        return [record["values"] for record in results]

    order_status_values = get_distinct_values("MaintenanceWorkOrder", "order_status")
    maintenance_type_values = get_distinct_values("MaintenanceWorkOrder", "maintenance_type")
    fault_category_values = get_distinct_values("MachineFault", "fault_category")

    schema = f"""
# Node Labels and Properties
(:MaintenanceWorkOrder {{work_order_id: 'INTEGER', maintenance_type: 'STRING' /* one of: {maintenance_type_values} */, order_status: 'STRING' /* one of: {order_status_values} */, planned_date: 'DATE', actual_finish_date: 'DATE'}})
(:Equipment {{sap_equipment_number: 'STRING'}})
(:MachineDowntimeEvent {{event_start_datetime: 'DATETIME', downtime_in_minutes: 'FLOAT'}})
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
(:MachineDowntimeEvent)-[:PRECEDES]->(:MachineDowntimeEvent)
"""
    return schema
# --- END: Self-Contained Schema Builder Logic ---

graph_schema = build_enriched_schema()

# Curated examples to teach specific, complex patterns
cypher_examples = [
    {
        "question": "What was the single most frequent fault description?",
        "query": """MATCH (f:MachineFault)<-[:DUE_TO_FAULT]-(d:MachineDowntimeEvent)
                    RETURN f.fault_description AS fault, COUNT(d) AS frequency
                    ORDER BY frequency DESC LIMIT 1;""",
    },
    {
        "question": "Find the longest cascading failure chain.",
        "query": """MATCH path = (root_cause:MachineDowntimeEvent)-[:PRECEDES*]->(downstream_event:MachineDowntimeEvent)
                    WHERE NOT ()-[:PRECEDES]->(root_cause)
                    WITH path, length(path) AS len
                    ORDER BY len DESC LIMIT 1
                    UNWIND nodes(path) AS event
                    MATCH (event)<-[:RECORDED_DOWNTIME_EVENT]-(m:Machine)
                    RETURN m.machine_description AS machine, event.event_start_datetime AS time
                    ORDER BY time;""",
    },
    {
        "question": "Are there any open work orders that are past their planned date?",
        "query": """MATCH (wo:MaintenanceWorkOrder)
                    WHERE wo.order_status IN ["In Progress", "Not Started", "Incomplete"] AND wo.planned_date < date()
                    RETURN wo.work_order_id, wo.work_order_description, wo.planned_date;""",
    }
]

# A new, more forceful and rule-driven prompt template
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j developer. Your ONLY task is to write a single, syntactically correct Cypher query to answer the user's question.

You MUST follow these strict rules:
1.  **ROOT CAUSE ANALYSIS:** For any question about "root cause", "cascading failure", "led to", or a sequence of events, you MUST traverse backwards using the `[:PRECEDES]` relationship. For example: `(cause)-[:PRECEDES*]->(effect)`.
2.  **COUNTING FREQUENCY:** To find the "frequency" or "number of times" a fault occurs, you MUST count the `(:MachineDowntimeEvent)` nodes connected to that fault, not the `(:MachineFault)` nodes themselves.
3.  **USE PROVIDED VALUES:** When a property has a comment listing possible values (e.g., `/* one of: ... */`), you MUST use the values from that list when filtering. Do not guess other values.
4.  **DATE CONVERSION:** You MUST convert `DATE` properties to `DATETIME` using `datetime(toString(date_property))` before adding a `duration`.
5.  **SCHEMA ADHERENCE:** Use ONLY the nodes, relationships, and properties provided in the Schema. Do not hallucinate any others.

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
            database=os.getenv("NEO4J_DATABASE", "neo4j")
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
            final_answer = result.get("result", "An error occurred or no data was found.")
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred while processing the query.", str(e)
