import streamlit as st
import pandas as pd
from datetime import date, timedelta
from cypher_chain import Neo4jLLMConnector
from db_connector import db_conn

# --- START: Advanced Analytics Functions ---
# These functions are now updated to accept a start_date and end_date

def calculate_mtbf(location_name, start_date, end_date):
    """
    Calculates MTBF for machines in a location within a specific date range.
    """
    query = """
    MATCH (l:Location)<-[:FALLS_UNDER]-(m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)
    WHERE l.location_name CONTAINS $location_name
      AND d.event_start_datetime >= datetime($start_date)
      AND d.event_start_datetime < datetime($end_date) + duration({days: 1})
    RETURN m.machine_description AS machine, d.event_start_datetime AS start_time, d.event_end_datetime AS end_time
    ORDER BY m.machine_description, start_time
    """
    params = {'location_name': location_name, 'start_date': str(start_date), 'end_date': str(end_date)}
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

def analyze_costliest_faults(start_date, end_date):
    """
    Finds top faults by downtime within a specific date range and classifies them.
    """
    query = """
    MATCH (m:Machine)-[:RECORDED_DOWNTIME_EVENT]->(d:MachineDowntimeEvent)-[:DUE_TO_FAULT]->(f:MachineFault)
    WHERE d.event_start_datetime >= datetime($start_date)
      AND d.event_start_datetime < datetime($end_date) + duration({days: 1})
    RETURN m.machine_description AS machine, 
           f.fault_description AS fault, 
           f.fault_category AS category, 
           d.downtime_in_minutes AS downtime
    """
    params = {'start_date': str(start_date), 'end_date': str(end_date)}
    results = db_conn.run_query(query, parameters=params)
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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Natural Language Query", "Advanced Analytics"])


# --- Page 1: Natural Language Query ---
if page == "Natural Language Query":
    st.title("🤖 Natural Language Querying with Neo4j")
    # This section remains unchanged
    # ... (Your existing LLM chat code) ...


# --- Page 2: Advanced Analytics ---
elif page == "Advanced Analytics":
    st.title("📈 Advanced Analytics Dashboard")

    # --- NEW: Date Range Selector ---
    st.header("Select Analysis Period")
    # Set default dates for the date pickers
    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(weeks=4)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start_date)
    with col2:
        end_date = st.date_input("End Date", value=default_end_date)
    
    st.divider()

    # --- MTBF Calculator ---
    st.header("Mean Time Between Failures (MTBF) Calculator")
    locations = ["Line 5", "Line 7"]
    selected_location = st.selectbox("Select a location:", locations)
    if st.button("Calculate MTBF"):
        with st.spinner(f"Calculating MTBF for {selected_location}..."):
            mtbf_df = calculate_mtbf(selected_location, start_date, end_date)
            if mtbf_df.empty:
                st.warning("No sufficient downtime data found for this period.")
            else:
                st.dataframe(mtbf_df, hide_index=True)

    st.divider()

    # --- Costliest Faults Analysis ---
    st.header("Costliest Faults Analysis")
    if st.button("Analyze Downtime"):
        with st.spinner("Analyzing costliest faults and loss types..."):
            top_faults_df, split_df = analyze_costliest_faults(start_date, end_date)
            
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

