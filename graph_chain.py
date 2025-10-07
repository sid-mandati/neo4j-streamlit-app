import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Updated schema for the smaller graph, for fast startup.
# This version removes SensorTag, PerformanceSnapshot, Sku, and QualityTest nodes.
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
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

    def ask(self, question):
        """
        Takes a user's question and returns the generated Cypher and the final answer.
        """
        try:
            result = self.chain.invoke({"query": question})
            
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "Query not generated.")
            final_answer = result.get("result", "Could not find an answer.")
            
            return cypher_query, final_answer
        except Exception as e:
            return "An error occurred", str(e)
