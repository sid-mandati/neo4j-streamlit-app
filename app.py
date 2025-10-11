from flask import Flask, render_template_string, request, jsonify
from cypher_chain import Neo4jLLMConnector
import os
import pandas as pd

app = Flask(__name__)

try:
    # Initialization should now be very fast.
    connector = Neo4jLLMConnector()
except Exception as e:
    print(f"FATAL: Failed to initialize Neo4jLLMConnector. Check credentials. Error: {e}")
    connector = None

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
        .message { margin-bottom: 15px; display: flex; flex-direction: column; max-width: 90%; word-wrap: break-word; }
        .user-message { align-self: flex-end; background-color: #0084ff; color: white; border-radius: 18px 18px 5px 18px; padding: 10px 15px; }
        .bot-message { align-self: flex-start; background-color: #e5e5ea; color: black; border-radius: 18px 18px 18px 5px; padding: 10px 15px; }
        #input-container { border-top: 1px solid #ddd; padding: 20px; display: flex; }
        #user-input { flex-grow: 1; border: 1px solid #ccc; border-radius: 18px; padding: 10px 15px; font-size: 16px; outline: none; }
        #send-button { background-color: #0084ff; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; margin-left: 10px; cursor: pointer; font-size: 20px; flex-shrink: 0; }
        .spinner { border: 4px solid rgba(0,0,0,0.1); width: 24px; height: 24px; border-radius: 50%; border-left-color: #0084ff; animation: spin 1s ease infinite; margin: auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; word-wrap: break-word; }
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

        function escapeHtml(text) {
            if (typeof text !== 'string') return text;
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
        }

        async function handleSend() {
            const question = userInput.value;
            if (!question.trim()) return;

            addMessage(escapeHtml(question), 'user');
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
                    botResponseHtml = `<p>Error: ${escapeHtml(data.error)}</p>`;
                } else if (data.final_answer) {
                    if (Array.isArray(data.final_answer) && data.final_answer.length > 0 && typeof data.final_answer[0] === 'object') {
                        botResponseHtml += '<table><thead><tr>';
                        Object.keys(data.final_answer[0]).forEach(key => botResponseHtml += `<th>${escapeHtml(key)}</th>`);
                        botResponseHtml += '</tr></thead><tbody>';
                        data.final_answer.forEach(row => {
                            botResponseHtml += '<tr>';
                            Object.values(row).forEach(value => botResponseHtml += `<td>${escapeHtml(value)}</td>`);
                            botResponseHtml += '</tr>';
                        });
                        botResponseHtml += '</tbody></table>';
                    } else {
                        botResponseHtml += `<p>${escapeHtml(data.final_answer)}</p>`;
                    }
                }
                
                if (data.cypher_query) {
                    botResponseHtml += '<h6>Generated Cypher Query:</h6><pre>' + escapeHtml(data.cypher_query) + '</pre>';
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

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    if not connector:
        return jsonify({"error": "Application not initialized. Check server logs."}), 500
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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
