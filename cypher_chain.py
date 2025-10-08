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

# Define Few-Shot Examples (with corrected type conversion for date math)
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
        "question": "Are there any overdue maintenance work orders?",
        "query": """MATCH (wo:MaintenanceWorkOrder)
                    WHERE wo.order_status IN ["In Progress", "Not Started"] AND wo.planned_date < date()
                    RETURN wo.work_order_id, wo.work_order_description, wo.planned_date
                    LIMIT 5;""",
    },
    {
        "question": "Did any downtime occur on a machine within 7 days after maintenance was completed on it?",
        "query": """MATCH (wo:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(e:Equipment)-[:MAPS_TO]->(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
                    WHERE d.event_start_datetime > datetime(wo.actual_finish_date) 
                      AND d.event_start_datetime < datetime(wo.actual_finish_date) + duration({days: 7})
                    RETURN m.machine_description, wo.work_order_description, d.event_start_datetime
                    LIMIT 5;""",
    }
]

# Define the Custom Prompt Template (with a new rule for type conversion)
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query developer. Your ONLY task is to write a single, syntactically correct Cypher query to answer the user's question. DO NOT add any text before or after the query.

You must follow these strict rules:
1.  **Use ONLY the nodes, relationships, and properties provided in the Schema.**
2.  **Follow the graph structure.** Do not create paths that do not exist.
3.  **Handle DATE to DATETIME conversion.** Properties of type `DATE` (like `actual_finish_date`) MUST be converted to a `DATETIME` using `datetime()` before you can add a `duration` to them. For example: `datetime(wo.actual_finish_date) + duration({days: 7})`.
4.  **To correlate maintenance with downtime**, use the path: `(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)-[:MAPS_TO]->(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)`.
5.  **Use provided values.** When a property has a comment listing possible values, you MUST use those values.
6.  **Count events, not nodes.** To find the "frequency" of a fault, you MUST count `MachineDowntimeEvent` nodes.
7.  **Always return properties.** Do not return entire nodes.

Schema:
{schema}
---
Here are some examples of questions and their correct Cypher queries:
{examples}
---
The question is:
{question}
"""

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
            result = self.chain.invoke({"query": question, "examples": cypher_examples})
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "Query not generated.")
            final_answer = result.get("result", "Could not find an answer.")
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred", str(e)
