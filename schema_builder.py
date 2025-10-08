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

    # Build the schema string with the fetched values included as comments
    schema = f"""
# Node Labels and Properties
(:MaintenanceWorkOrder {{
    work_order_id: 'INTEGER', 
    work_order_description: 'STRING', 
    maintenance_type: 'STRING' /* one of: {maintenance_type_values} */, 
    order_status: 'STRING' /* one of: {order_status_values} */,
    actual_finish_date: 'DATE'
}})
(:Equipment {{sap_equipment_number: 'STRING', sap_equipment_description: 'STRING'}})
(:MachineDowntimeEvent {{event_start_datetime: 'DATETIME', downtime_in_minutes: 'FLOAT'}})
(:Machine {{machine_id: 'STRING', machine_description: 'STRING'}})
(:Location {{location_id: 'STRING', location_name: 'STRING'}})
(:MachineFault {{
    plant_line_machine_fault_code_id: 'STRING', 
    fault_description: 'STRING', 
    fault_category: 'STRING' /* one of: {fault_category_values} */
}})

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
    return schema
