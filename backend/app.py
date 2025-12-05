# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Needed to allow React to talk to Flask
from werkzeug.utils import secure_filename
import tempfile
import uuid
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Assuming your MathScanner class is in math_scanner.py (ensure it's in the same directory or importable)
from convert import MathScanner 

# --- Configuration ---
app = Flask(__name__)
# Enable CORS for cross-origin requests (essential when React and Flask run on different ports)
CORS(app) 

# Instantiate the MathScanner class once when the server starts
try:
    scanner = MathScanner() 
except Exception as e:
    print(f"FATAL: Failed to initialize MathScanner. Is pix2tex installed? Error: {e}")
    scanner = None
# --- API Route ---

@app.route('/api/process', methods=['POST'])
def process_image():
    # 1. Check for the file
    if 'image' not in request.files:
        return jsonify({'success': False, 'error_message': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error_message': 'No selected file'}), 400
    
    if not scanner:
        return jsonify({'success': False, 'error_message': 'MathScanner model failed to load on server startup.'}), 503

    # 2. Save the file securely to a temporary location
    # Use a unique name to prevent conflicts
    unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, unique_filename)
    
    try:
        file.save(temp_path)
        print(f"Saved uploaded file temporarily to: {temp_path}")

        # 3. Run the pipeline
        final_latex = scanner.run_pipeline(temp_path)
        
        # 4. Respond with JSON data
        return jsonify({
            'success': True,
            'full_latex_document': final_latex
        })
        
    except ValueError as e:
        # Catch errors explicitly raised by your pipeline (e.g., file not found, but unlikely here)
        return jsonify({'success': False, 'error_message': f'Pipeline failed during processing: {str(e)}'}), 500
        
    except Exception as e:
        # Catch unexpected errors
        return jsonify({'success': False, 'error_message': f'Unexpected server error: {str(e)}'}), 500
        
    finally:
        # 5. Clean up the temporary file (Crucial!)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")

if __name__ == '__main__':
    # Flask runs on port 5000 by default. React runs on 3000.
    app.run(debug=True, port=5000)