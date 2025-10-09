from flask import Flask, render_template_string, request, jsonify
from cypher_chain import Neo4jLLMConnector
import os
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Initialize our LangChain connector. This happens only once when the app starts.
# It's critical that this is defined in the global scope for efficiency.
try:
    print("Initializing Neo4jLLMConnector...")
    connector = Neo4jLLMConnector()
    print("Initialization complete.")
except Exception as e:
    print(f"FATAL: Failed to initialize Neo4jLLMConnector. Error: {e}")
    connector = None

# --- HTML, CSS, & JavaScript Template ---
# This contains the entire user interface for the chat application.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neo4j Chat</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
        #app-container { width: 100%; max-width: 800px; background: #fff; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; height: 90vh; }
        header { padding: 20px; border-bottom: 1px solid #ddd; font-size: 24px; font-weight: bold; color: #333; text-align: center; }
        #chat-window { flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { display: flex; flex-direction: column; max-width: 85%; }
        .user-message { align-self: flex-end; }
        .user-message .content { background-color: #0084ff; color: white; border-radius: 18px 18px 5px 18px; padding: 10px 15px; }
        .bot-message { align-self: flex-start; }
        .bot-message .content { background-color: #e5e5ea; color: black; border-radius: 18px 18px 18px 5px; padding: 10px 15px; }
        #input-container { border-top: 1px solid #ddd; padding: 20px; display: flex; }
        #user-input { flex-grow: 1; border: 1px solid #ccc; border-radius: 18px; padding: 10px 15px; font-size: 16px; outline: none; transition: border-color 0.2s; }
        #user-input:focus { border-color: #0084ff; }
        #send-button { background-color: #0084ff; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; margin-left: 10px; cursor: pointer; font-size: 20px; display: flex; justify-content: center; align-items: center; transition: background-color 0.2s; }
        #send-button:hover { background-color: #006bcf; }
        .spinner { border: 4px solid rgba(0,0,0,0.1); width: 24px; height: 24px; border-radius: 50%; border-left-color: #0084ff; animation: spin 1s ease infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        h6 { margin: 15px 0 5px 0; color: #555; }
    </style>
</head>
<body>
    <div id="app-container">
        <header>🤖 Natural Language Querying with Neo4j</header>
        <div id="chat-window"></div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="Ask a question about your plant data..." autofocus>
            <button id="send-button" aria-label="Send Message">→</button>
        </div>
    </div>

    <script>
        const chatWindow = document.getElementById('chat-window');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        function addMessage(htmlContent, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            contentDiv.innerHTML = htmlContent;
            
            messageDiv.appendChild(contentDiv);
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return messageDiv;
        }

        async function handleSend() {
            const question = userInput.value;
            if (!question.trim()) return;

            addMessage(question, 'user');
            userInput.value = '';
            
            // Add a temporary bot message with a spinner
            const tempBotMessage = addMessage('<div class="spinner"></div>', 'bot');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

                const data = await response.json();
                
                let botResponseHtml = '';
                if (data.error) {
                     botResponseHtml = `<p>Error: ${data.error}</p>`;
                } else if (data.final_answer) {
                    if (Array.isArray(data.final_answer) && data.final_answer.length > 0 && typeof data.final_answer[0] === 'object') {
                        // Render as a table
                        botResponseHtml += '<table><thead><tr>';
                        Object.keys(data.final_answer[0]).forEach(key => {
                            botResponseHtml += `<th>${key.replace(/_/g, ' ')}</th>`; // Format headers
                        });
                        botResponseHtml += '</tr></thead><tbody>';
                        data.final_answer.forEach(row => {
                            botResponseHtml += '<tr>';
                            Object.values(row).forEach(value => {
                                botResponseHtml += `<td>${value}</td>`;
                            });
                            botResponseHtml += '</tr>';
                        });
                        botResponseHtml += '</tbody></table>';
                    } else {
                        // Render as simple text
                        botResponseHtml += `<p>${data.final_answer}</p>`;
                    }
                } else {
                    botResponseHtml = `<p>Sorry, I could not find an answer.</p>`;
                }
                
                if (data.cypher_query) {
                    botResponseHtml += '<h6>Generated Cypher Query:</h6><pre>' + data.cypher_query.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</pre>';
                }
                
                // Update the bot message with the actual response
                tempBotMessage.querySelector('.content').innerHTML = botResponseHtml;

            } catch (error) {
                console.error('Fetch Error:', error);
                tempBotMessage.querySelector('.content').innerHTML = '<p>Sorry, an error occurred. Please check the server logs for more details.</p>';
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
        return jsonify({"error": "The application is not initialized correctly. Check the server logs."}), 500
        
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        cypher_query, final_answer = connector.ask(question)
        return jsonify({
            "cypher_query": cypher_query,
            "final_answer": final_answer
        })
    except Exception as e:
        print(f"Error during ask: {e}")
        return jsonify({"error": "An internal error occurred while processing the question."}), 500

# This is required for Google App Engine which uses a production web server like Gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
