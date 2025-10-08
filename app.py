import streamlit as st
import pandas as pd
from cypher_chain import Neo4jLLMConnector
from db_connector import db_conn

# --- START: Advanced Analytics Functions ---

def calculate_mtbf(location_name, weeks=2):
    """
    Calculates the Mean Time Between Failures (MTBF) for all machines in a given location.
    """
    query = """
    MATCH (l:Location)<-[:FALLS_UNDER]-(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
    WHERE l.location_name CONTAINS $location_name
      AND d.event_start_datetime >= datetime() - duration({weeks: $weeks})
    RETURN m.machine_description AS machine, d.event_start_datetime AS start_time, d.event_end_datetime AS end_time
    ORDER BY m.machine_description, start_time
    """
    params = {'location_name': location_name, 'weeks': weeks}
    results = db_conn.run_query(query, parameters=params)
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results)
    df['start_time'] = pd.to_datetime(df['start_time'].apply(lambda x: x.to_native()))
    df['end_time'] = pd.to_datetime(df['end_time'].apply(lambda x: x.to_native()))
    mtbf_results = []
    for machine_name, group in df.groupby('machine'):
        group = group.sort_values('start_time').reset_index()
        if len(group) < 2: continue
        time_between_failures = group['start_time'].shift(-1) - group['end_time']
        time_between_failures = time_between_failures.dropna()
        if not time_between_failures.empty:
            mean_time_between_failures = time_between_failures.mean()
            mtbf_hours = mean_time_between_failures.total_seconds() / 3600
            mtbf_results.append({
                'Machine': machine_name, 'MTBF (Hours)': round(mtbf_hours, 2), 'Number of Failures': len(group)
            })
    return pd.DataFrame(mtbf_results).sort_values(by='MTBF (Hours)', ascending=False)

def analyze_costliest_faults(weeks=4):
    """
    Finds the top faults by downtime and classifies them as Breakdown vs. Operational Loss.
    """
    query = """
    MATCH (m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(f:MachineFault)
    WHERE d.event_start_datetime >= datetime() - duration({weeks: $weeks})
    RETURN m.machine_description AS machine, 
           f.fault_description AS fault, 
           f.fault_category AS category, 
           d.downtime_in_minutes AS downtime
    """
    results = db_conn.run_query(query, parameters={'weeks': weeks})
    if not results: return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(results)

    top_faults = df.groupby(['machine', 'fault', 'category']).agg(
        total_downtime=('downtime', 'sum'),
        occurrences=('downtime', 'count')
    ).reset_index().sort_values(by='total_downtime', ascending=False).head(5)

    df['loss_type'] = df['category'].apply(lambda x: 'Breakdown' if x in ['Mechanical', 'Electrical'] else 'Operational Loss')
    
    split_analysis = df.groupby('loss_type').agg(
        total_downtime=('downtime', 'sum')
    ).reset_index()
    
    return top_faults, split_analysis

# --- END: Advanced Analytics Functions ---


st.set_page_config(layout="wide")

# --- Sidebar for Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Natural Language Query", "Advanced Analytics"])


# --- Page 1: Natural Language Query ---
if page == "Natural Language Query":
    st.title("🤖 Natural Language Querying with Neo4j")

    try:
        if 'connector' not in st.session_state:
            with st.spinner("Connecting to services and building dynamic schema..."):
                st.session_state.connector = Neo4jLLMConnector()
    except Exception as e:
        st.error(f"Failed to initialize. Check your credentials. Error: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your plant data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating Cypher query and finding answer..."):
                cypher_query, final_answer = st.session_state.connector.ask(prompt)
                
                with st.expander("View Generated Cypher Query"):
                    st.code(cypher_query, language="cypher")
                
                if isinstance(final_answer, list) and final_answer and all(isinstance(item, dict) for item in final_answer):
                    st.dataframe(pd.DataFrame(final_answer))
                else:
                    st.markdown(final_answer)
        
        full_response = f"**Answer:** {final_answer}\n\n**Generated Query:**\n```cypher\n{cypher_query}\n```"
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Page 2: Advanced Analytics ---
elif page == "Advanced Analytics":
    st.title("📈 Advanced Analytics Dashboard")

    # --- MTBF Calculator ---
    st.header("Mean Time Between Failures (MTBF) Calculator")
    locations = ["Line 5", "Line 7"]
    selected_location = st.selectbox("Select a location:", locations)
    if st.button("Calculate MTBF for Last 2 Weeks"):
        with st.spinner(f"Calculating MTBF for {selected_location}..."):
            mtbf_df = calculate_mtbf(selected_location, weeks=2)
            if mtbf_df.empty:
                st.warning("No sufficient downtime data found.")
            else:
                st.dataframe(mtbf_df, hide_index=True)

    st.divider()

    # --- Costliest Faults Analysis ---
    st.header("Costliest Faults Analysis")
    if st.button("Analyze Downtime for Last 4 Weeks"):
        with st.spinner("Analyzing costliest faults and loss types..."):
            top_faults_df, split_df = analyze_costliest_faults(weeks=4)
            
            st.subheader("Top 5 Machine-Fault Combinations by Downtime")
            if top_faults_df.empty:
                st.warning("No downtime data found for this period.")
            else:
                st.dataframe(top_faults_df, hide_index=True)

            st.subheader("Downtime Split: Breakdown vs. Operational Loss")
            if not split_df.empty:
                st.vega_lite_chart(split_df, {
                    'mark': {'type': 'arc', 'tooltip': True},
                    'encoding': {
                        'theta': {'field': 'total_downtime', 'type': 'quantitative'},
                        'color': {'field': 'loss_type', 'type': 'nominal'},
                    },
                })
