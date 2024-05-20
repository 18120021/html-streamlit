import socketserver
import threading
import webbrowser
import subprocess
import time
import os
from http.server import SimpleHTTPRequestHandler

# Define the port for the HTTP server
PORT = 8001
STREAMLIT_PORT = 8502

# Define the handler to serve the HTML file
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

# Function to start the HTTP server
def start_http_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving HTML on port {PORT}")
        httpd.serve_forever()

# Function to start the Streamlit app
def start_streamlit():
    os.system('streamlit run /Users/Nauroha/Downloads/TA/main2.py.py --server.port {STREAMLIT_PORT}')

# Start the HTTP server in a separate thread
threading.Thread(target=start_http_server).start()

# Start the Streamlit app
start_streamlit()

# Give the server a moment to start
time.sleep(1)

# Open the HTML file in the default web browser
webbrowser.open(f'http://localhost:{PORT}')

print("HTTP server started. Open the HTML file in your browser and click the button to start the Streamlit app.")
