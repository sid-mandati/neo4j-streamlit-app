import os
from neo4j import GraphDatabase

def get_distinct_values(session, node_label, property_name):
    """Helper function to run a DISTINCT query using session.run()."""
    query = f"MATCH (n:{node_label}) WHERE n.{property_name} IS NOT NULL RETURN DISTINCT n.{property_name} AS values"
    result = session.run(query)
    return [record["values"] for record in result]

def build_enriched_schema():
    """
    Connects to Neo4j, fetches distinct values for key properties,
    and returns a formatted schema string.
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
(:MaintenanceWorkOrder {{
    work_order_id: 'INTEGER', 
    work_order_description: 'STRING', 
    maintenance_type: 'STRING' /* one of: {maintenance_type_values} */, 
    order_status: 'STRING' /* one of: {order_status_values
