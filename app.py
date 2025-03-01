import logging
from flask import Flask, render_template, request, jsonify, Response
import asyncio
from dev.upload_to_vectordb import VectorDBUploader
import time
import signal
import json

app = Flask(__name__)

# Set up logging to a file and console
logger = logging.getLogger('VectorDBUploader')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log', mode='w')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global variable to control cancellation
cancel_flag = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global cancel_flag
    if request.method == 'POST':
        cancel_flag = False
        sitemap_url = request.form.get('sitemap_url')
        use_sitemap = request.form.get('use_sitemap') == 'on'
        format_content = request.form.get('format_content') == 'on'
        source = request.form.get('source')
        info_type = request.form.get('infoType')
        
        # Get line pairs from the request
        line_pairs = request.form.get('line_pairs')
        if line_pairs:
            line_pairs = json.loads(line_pairs)  # Convert the JSON string back to a list

        # Get line numbers from the request
        line_numbers_input = request.form.get('line_numbers')
        line_numbers = []
        if line_numbers_input:
            try:
                line_numbers = [int(num.strip()) for num in line_numbers_input.split(',')]
            except ValueError:
                return jsonify({"error": "Invalid line numbers format."}), 400

        # Pass line_pairs and line_numbers to the uploader
        asyncio.run(run_uploader(sitemap_url, use_sitemap, format_content, source, line_pairs, info_type, line_numbers))
        return jsonify({"status": "Running"})

    return render_template('index.html')

@app.route('/cancel', methods=['POST'])
def cancel():
    global cancel_flag
    cancel_flag = True
    logger.info("Cancellation requested - stopping all operations")
    return jsonify({"status": "Cancelled"})

@app.route('/logs')
def stream_logs():
    def generate():
        with open('app.log', 'r') as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    time.sleep(0.1)  # Small delay to prevent busy-waiting
    return Response(generate(), mimetype='text/event-stream')

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An error occurred: {e}")
    return jsonify({"error": "An internal error occurred"}), 500

async def run_uploader(sitemap_url, use_sitemap, format_content, source, line_pairs, info_type, line_numbers):
    global cancel_flag
    cancel_flag = False  # Reset cancel flag at start
    
    try:
        uploader = VectorDBUploader(
            url=sitemap_url, 
            sitemap=use_sitemap, 
            format_content=format_content, 
            source=source,
            line_pairs=line_pairs,
            info_type=info_type,
            line_numbers=line_numbers,
            cancel_check=lambda: cancel_flag
        )
        await uploader.run()
    except asyncio.CancelledError:
        logger.info("Task was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        raise
    finally:
        if cancel_flag:
            logger.info("Upload process was cancelled")

if __name__ == '__main__':
    app.run(debug=True)
