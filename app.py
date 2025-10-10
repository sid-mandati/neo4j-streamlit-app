from flask import Flask
import os
import pandas as pd
# Import our new database connector
from db_connector import db_conn 

app = Flask(__name__)

@app.route('/')
def index():
    """
    Runs a test query and displays the result or an error message.
    """
    # Test query to count the number of Machine nodes
    query = "MATCH (m:Machine) RETURN count(m) AS total_machines"
    
    try:
        results = db_conn.run_query(query)
        if results:
            # Display the result in a simple table
            df = pd.DataFrame(results)
            return f"<h1>Database Connection Successful!</h1><p>The application successfully connected to your Neo4j database and ran a query.</p>{df.to_html(index=False)}"
        else:
            return "<h1>Database Connection Successful!</h1><p>Query ran, but returned no results.</p>", 200

    except Exception as e:
        # Display an error if the connection fails
        return f"<h1>Database Connection Failed</h1><p>Could not connect to Neo4j. Check your environment variables in Render.</p><p>Error: {e}</p>", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
