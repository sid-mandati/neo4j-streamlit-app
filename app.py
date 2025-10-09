import os
from flask import Flask, render_template_string, request, jsonify
from google.cloud import secretmanager

# --- START: Secure Secret Loading ---
# This block runs only once when the application starts.

def access_secret_version(project_id, secret_id, version_id="latest"):
    """
    Access the payload for the given secret version and return it.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Get the Project ID from the environment (automatically set by App Engine)
project_id = os.environ.get('GCP_PROJECT')

if project_id:
    print("Loading secrets from Google Cloud Secret Manager...")
    try:
        # Fetch secrets and set them as environment variables
        os.environ['OPENAI_API_KEY'] = access_secret_version(project_id, 'openai-api-key')
        os.environ['NEO4J_URI'] = access_secret_version(project_id, 'neo4j-uri')
        os.environ['NEO4J_USER'] = access_secret_version(project_id, 'neo4j-user')
        os.environ['NEO4J_PASSWORD'] = access_secret_version(project_id, 'neo4j-password')
        print("Secrets loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load secrets from Secret Manager. Error: {e}")
# --- END: Secure Secret Loading ---


# Now we can import our chain, which will use the environment variables we just set
from cypher_chain import Neo4jLLMConnector

app = Flask(__name__)

try:
    connector = Neo4jLLMConnector()
except Exception as e:
    print(f"FATAL: Failed to initialize Neo4jLLMConnector. Error: {e}")
    connector = None

# ... (The rest of your app.py file remains the same) ...

# --- HTML, CSS, and JavaScript Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neo4j Chat</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f0f2f5; }
        #app-container { max-width: 800px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; height: 85vh; }
        header { padding: 20px; border-bottom: 1px solid #ddd; font-size: 24px; font-weight: bold; color: #333; text-align: center; }
        #chat-window { flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; }
        .message { margin-bottom: 15px; display: flex; flex-direction: column; max-width: 80%; }
        .user-message { align-self: flex-end; background-color: #0084ff; color: white; border-radius: 18px 18px 5px 18px; padding: 10px 15px; }
        .bot-message { align-self: flex-start; background-color: #e5e5ea; color: black; border-radius: 18px 18px 18px 5px; padding: 10px 15px; }
        #input-container { border-top: 1px solid #ddd; padding: 20px; display: flex; }
        #user-input { flex-grow: 1; border: 1px solid #ccc; border-radius: 18px; padding: 10px 15px; font-size: 16px; outline: none; }
        #send-button { background-color: #0084ff; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; margin-left: 10px; cursor: pointer; font-size: 20px; }
        .spinner { border: 4px solid rgba(0,0,0,0.1); width: 24px; height: 24px; border-radius: 50%; border-left-color: #0084ff; animation: spin 1s ease infinite; margin: auto;}
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div id="app-container">
        <header>🤖 Natural Language Querying with Neo4j</header>
        <div id="chat-window"></div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="Ask a question about your plant data..." autofocus>
            <button id="send-button">➤</button>
        </div>
    </div>

    <script>
        const chatWindow = document.getElementById('chat-window');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        function addMessage(content, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.innerHTML = content;
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return messageDiv;
        }

        async function handleSend() {
            const question = userInput.value;
            if (!question.trim()) return;

            addMessage(question, 'user');
            userInput.value = '';
            
            const botMessageContainer = addMessage('<div class="spinner"></div>', 'bot');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();
                
                let botResponseHtml = '';
                if (data.error) {
                    botResponseHtml = `<p>Error: ${data.error}</p>`;
                } else if (data.final_answer) {
                    if (Array.isArray(data.final_answer) && data.final_answer.length > 0 && typeof data.final_answer[0] === 'object') {
                        botResponseHtml += '<table><thead><tr>';
                        Object.keys(data.final_answer[0]).forEach(key => botResponseHtml += `<th>${key}</th>`);
                        botResponseHtml += '</tr></thead><tbody>';
                        data.final_answer.forEach(row => {
                            botResponseHtml += '<tr>';
                            Object.values(row).forEach(value => botResponseHtml += `<td>${value}</td>`);
                            botResponseHtml += '</tr>';
                        });
                        botResponseHtml += '</tbody></table>';
                    } else {
                        botResponseHtml += `<p>${data.final_answer}</p>`;
                    }
                }
                
                if (data.cypher_query) {
                    botResponseHtml += '<h6>Generated Cypher Query:</h6><pre>' + data.cypher_query.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</pre>';
                }
                
                botMessageContainer.innerHTML = botResponseHtml;

            } catch (error) {
                botMessageContainer.innerHTML = '<p>Sorry, an error occurred while connecting to the server.</p>';
            }
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        sendButton.addEventListener('click', handleSend);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleSend();
            }
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    """Handles the API call from the frontend to ask a question."""
    if not connector:
        return jsonify({"error": "The application is not initialized correctly. Check server logs."}), 500
        
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        cypher_query, final_answer = connector.ask(question)
        return jsonify({"cypher_query": cypher_query, "final_answer": final_answer})
    except Exception as e:
        print(f"Error during ask: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
