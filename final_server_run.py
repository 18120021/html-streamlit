import socketserver
import threading
import webbrowser
import subprocess
import time
from http.server import SimpleHTTPRequestHandler

# Define the port for the HTTP server
PORT = 8001
STREAMLIT_PORT = 8502

# Define the handler to serve the HTML file and handle starting Streamlit
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'
            if self.path == '/start-streamlit':
                self.start_streamlit_app()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Streamlit app is starting...')
                return
        return SimpleHTTPRequestHandler.do_GET(self)

    def start_streamlit_app(self):
        if not is_streamlit_running():
            threading.Thread(target=start_streamlit, daemon=True).start()

# Function to check if Streamlit is already running
def is_streamlit_running():
    import requests
    try:
        response = requests.get(f'http://localhost:{STREAMLIT_PORT}')
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        return False
    return False

# Function to start the HTTP server
def start_http_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving HTML on port {PORT}")
        httpd.serve_forever()

# Function to start the Streamlit app
def start_streamlit():
    print("Starting Streamlit app...")
    process = subprocess.Popen(['streamlit', 'run', '/Users/Nauroha/Downloads/TA/main2.py', '--server.port', str(STREAMLIT_PORT)])
    process.communicate()

# Start the HTTP server in a separate thread
threading.Thread(target=start_http_server, daemon=True).start()

# Give the server a moment to start
time.sleep(1)

# Open the HTML file in the default web browser
webbrowser.open(f'http://localhost:{PORT}')

print("HTTP server started. Open the HTML file in your browser and click the button to start the Streamlit app.")