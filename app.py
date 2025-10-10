from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    """Serves a simple 'Hello, World!' message."""
    return "<h1>Hello, World!</h1><p>If you can see this, your Flask app is running correctly on Google App Engine.</p>"

if __name__ == '__main__':
    # This part is used for local testing, App Engine uses the entrypoint in app.yaml
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

