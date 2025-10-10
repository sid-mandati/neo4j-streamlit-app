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
    """
    Connects to Neo4j using db_conn, fetches distinct values, and returns a schema string.
    """
    def get_distinct_values(node_label, property_name):
        query = f"MATCH (n:{node_label}) WHERE n.{property_name} IS NOT NULL RETURN DISTINCT n.{property_name} AS values"
        results = db_conn.run_query(query)
        return [record["values"] for record in results]

    order_status_values = get_distinct_values("MaintenanceWorkOrder", "order_status")
    maintenance_type_values = get_distinct_values("MaintenanceWorkOrder", "maintenance_type")
    fault_category_values = get_distinct_values("MachineFault", "fault_category")

    schema = f"""
# Node Labels and Properties
(:MaintenanceWorkOrder {{work_order_id: 'INTEGER', maintenance_type: 'STRING' /* one of: {maintenance_type_values} */, order_status: 'STRING' /* one of: {order_status_values} */, actual_finish_date: 'DATE'}})
(:Equipment {{sap_equipment_number: 'STRING', sap_equipment_description: 'STRING'}})
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
"""
    return schema
# --- END: Self-Contained Schema Builder Logic ---

graph_schema = build_enriched_schema()

cypher_examples = [
    {"question": "How many machines are there?", "query": "MATCH (m:Machine) RETURN count(m);"},
    {"question": "Did any downtime occur on a machine within 7 days after maintenance?", "query": "MATCH (wo:MaintenanceWorkOrder)-[:PERFORMED_ON_EQUIPMENT]->(e:Equipment)-[:MAPS_TO]->(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent) WHERE d.event_start_datetime > datetime(wo.actual_finish_date) AND d.event_start_datetime < datetime(wo.actual_finish_date) + duration({days: 7}) RETURN m.machine_description, wo.work_order_description LIMIT 5;"},
]

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j developer. Write a Cypher query to answer the user's question.

You must follow these strict rules:
1. Use ONLY the nodes, relationships, and properties in
