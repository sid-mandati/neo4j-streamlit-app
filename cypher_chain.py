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

# Define Few-Shot Examples
cypher_examples = [
    {"question": "How many machines are there?", "query": "MATCH (m:Machine) RETURN count(m);"},
    {"question": "Did any downtime occur on a machine within 7 days after maintenance was completed on it?", "query": "MATCH (wo:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(e:Equipment)-[:MAPS_TO]->(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent) WHERE d.event_start_datetime > datetime(wo.actual_finish_date) AND d.event_start_datetime < datetime(wo.actual_finish_date) + duration({days: 7}) RETURN m.machine_description, wo.work_order_description LIMIT 5;"},
]

# Define the Custom Prompt Template
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j developer. Write a Cypher query to answer the user's question.

You must follow these strict rules:
1. Use ONLY the nodes, relationships, and properties in the provided Schema.
2. To correlate maintenance with downtime, use the path: `(:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(:Equipment)-[:MAPS_TO]->(:Machine)-[:RECORDED_DOWNTIME_EVENT]->(:MachineDowntimeEvent)`.
3. Handle DATE to DATETIME conversion using the `datetime()` function before adding a duration.

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
