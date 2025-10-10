from flask import Flask
import os
import pandas as pd
from db_connector import db_conn 

app = Flask(__name__)

@app.route('/')
def index():
    """
    Runs a test query and displays the result or an error message.
    """
    query = "MATCH (m:Machine) RETURN count(m) AS total_machines"
    
    try:
        results = db_conn.run_query(query)
        if results and results[0]['total_machines'] > 0:
            df = pd.DataFrame(results)
            return f"<h1>Database Connection Successful!</h1><p>The application connected to the correct database and found data.</p>{df.to_html(index=False)}"
        else:
            return "<h1>Database Connection Successful!</h1><p>Query ran, but returned no results. Verify data exists in the 'neo4j' database.</p>", 200

    except Exception as e:
        return f"<h1>Database Connection Failed</h1><p>Could not connect to Neo4j. Check your environment variables in Render.</p><p>Error: {e}</p>", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
