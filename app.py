import streamlit as st
import pandas as pd
from cypher_chain import Neo4jLLMConnector

st.set_page_config(layout="wide")
st.title("🤖 Natural Language Querying with Neo4j")

try:
    if 'connector' not in st.session_state:
        with st.spinner("Connecting to services and building dynamic schema... Please wait."):
            st.session_state.connector = Neo4jLLMConnector()
except Exception as e:
    st.error(f"Failed to initialize. Check your credentials in the secrets manager. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message.get("display_content"), pd.DataFrame):
            st.dataframe(message["display_content"])
        else:
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
                df_answer = pd.DataFrame(final_answer)
                st.dataframe(df_answer)
                st.session_state.messages.append({"role": "assistant", "content": "Here is the data in a table:", "display_content": df_answer})
            else:
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": str(final_answer)})
