# app.py (Revised)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import uuid
import os
import sys
import base64
import cv2
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Assuming your MathScanner class is in math_scanner.py (ensure it's in the same directory or importable)
from convert import MathScanner 

# --- Global State & Config ---
app = Flask(__name__)
CORS(app) 
scanner = MathScanner(use_model=True) # Assuming the model loads
JOBS = {} # Dictionary to store job state: {job_id: {status: '...', temp_path: '...', steps: [...]}}

# Temporary directory for file uploads and intermediate images
TEMP_PROCESSING_DIR = os.path.join(tempfile.gettempdir(), "mathscanner_jobs")
if not os.path.exists(TEMP_PROCESSING_DIR):
    os.makedirs(TEMP_PROCESSING_DIR)

# Utility to convert OpenCV image to Base64 (needed for reporting intermediate images)
def cv2_to_base64(img):
    """Converts a numpy array image (cv2 format) to a base64 encoded string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# --- API Routes ---
@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'success': False, 'error_message': 'No image file uploaded'}), 400

    file = request.files['image']
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_PROCESSING_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    original_filename = secure_filename(file.filename)
    temp_path = os.path.join(job_dir, original_filename)
    
    try:
        file.save(temp_path)
        
        # Initialize job state
        JOBS[job_id] = {
            'status': 'uploaded',
            'temp_path': temp_path,
            'result': None,
            'steps': { # Initialize step states and image holders
                'preprocessing': {'status': 'pending', 'image': None},
                'ocrRecognition': {'status': 'pending', 'image': None},
                'segmentation': {'status': 'pending', 'image': None},
                'modelInference': {'status': 'pending', 'image': None},
                'reassembly': {'status': 'pending', 'image': None},
                'validationOutput': {'status': 'pending', 'image': None},
            }
        }
        
        print(f"Job {job_id}: File saved to {temp_path}")
        return jsonify({'success': True, 'job_id': job_id}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error_message': f'File upload failed: {str(e)}'}), 500


@app.route('/api/process/<job_id>', methods=['POST'])
def process_image(job_id):
    if job_id not in JOBS:
        return jsonify({'success': False, 'error_message': 'Job ID not found'}), 404
        
    job_data = JOBS[job_id]
    image_path = job_data['temp_path']
    
    if not scanner:
        job_data['status'] = 'failed'
        return jsonify({'success': False, 'error_message': 'MathScanner model failed to load.'}), 503

    try:
        job_data['status'] = 'processing'
        
        # --- Run Pipeline Stages Sequentially & Update State ---
        
        # 1. Skew Correction
        job_data['steps']['preprocessing']['status'] = 'processing'
        straight_img = scanner.correct_skew(image_path)
        job_data['steps']['preprocessing']['status'] = 'success'
        # Returns the straightened image (01e_straightened)
        job_data['steps']['preprocessing']['image'] = cv2_to_base64(straight_img) 


        # 2. OCR Recognition (Cleaning and Binarizing)
        job_data['steps']['ocrRecognition']['status'] = 'processing'
        binary, clean_bg = scanner.clean_image(straight_img) 
        job_data['steps']['ocrRecognition']['status'] = 'success'
        # Pass the 1-channel binary image directly to the improved utility
        job_data['steps']['ocrRecognition']['image'] = cv2_to_base64(binary)

        # 3. Segmentation (Merging Boxes)
        job_data['steps']['segmentation']['status'] = 'processing'
        # The MathScanner segment_regions internally draws bounding boxes on clean_bg 
        # (resulting in 03b_segmented_boxes_refined). We need to get that image.
        regions, segmented_img = scanner.segment_regions(binary, clean_bg)
        job_data['steps']['segmentation']['status'] = 'success'
        job_data['steps']['segmentation']['image'] = cv2_to_base64(segmented_img)
        
        
        # 4. Model Inference
        job_data['steps']['modelInference']['status'] = 'processing'
        # We don't have a single image for this step unless we stitch the crops
        latex_lines = scanner.image_to_latex(regions) 
        job_data['steps']['modelInference']['status'] = 'success'
        job_data['steps']['modelInference']['image'] = None # No single image for this step


        # 5. Final Reassembly
        job_data['steps']['reassembly']['status'] = 'processing'
        final_latex_doc = scanner.reassemble_document(latex_lines)
        job_data['steps']['reassembly']['status'] = 'success'
        
        # 6. Final Output (Meta status)
        job_data['steps']['validationOutput']['status'] = 'success'
        job_data['result'] = final_latex_doc
        job_data['status'] = 'complete'

        return jsonify({'success': True, 'message': 'Processing initiated. Poll /api/status/<job_id> for updates.'}), 202
    
    except Exception as e:
        job_data['status'] = 'failed'
        print(f"[ERROR] Job {job_id} failed: {e}")
        return jsonify({'success': False, 'error_message': f'Pipeline failed: {str(e)}'}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in JOBS:
        return jsonify({'success': False, 'error_message': 'Job ID not found'}), 404
        
    job_data = JOBS[job_id]
    # Return a simplified view of the job state for the client
    return jsonify({
        'status': job_data['status'],
        'steps': job_data['steps'],
        'result': job_data['result']
    }), 200

if __name__ == '__main__':
    # NOTE: Run Flask in a development environment for testing
    app.run(debug=True, port=5000)