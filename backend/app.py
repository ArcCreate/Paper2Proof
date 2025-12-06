from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import uuid
import os
import sys
import base64
import cv2
import threading
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert import MathScanner 

# Define a background worker function
def run_pipeline_background(job_id):
    job_data = JOBS[job_id]
    image_path = job_data['temp_path']
    
    try:
        # --- STEP 1: Skew Correction ---
        job_data['steps']['preprocessing']['status'] = 'processing'
        lines_vis, straight_img = scanner.correct_skew(image_path)
        job_data['steps']['preprocessing']['status'] = 'success'
        job_data['steps']['preprocessing']['images'] = [
            {'label': 'Detected Lines', 'src': cv2_to_base64(lines_vis)},
            {'label': 'Straightened Image', 'src': cv2_to_base64(straight_img)}
        ]

        # --- STEP 2: Cleaning ---
        job_data['steps']['ocrRecognition']['status'] = 'processing'
        cleaned_vis, binary_result = scanner.clean_image(straight_img)
        job_data['steps']['ocrRecognition']['status'] = 'success'
        job_data['steps']['ocrRecognition']['images'] = [
            {'label': 'Cleaned (Full Detail)', 'src': cv2_to_base64(cleaned_vis)}
        ]

        # --- STEP 3: Segmentation ---
        job_data['steps']['segmentation']['status'] = 'processing'
        dilation_vis, boxes_vis, regions = scanner.segment_regions(binary_result, straight_img)
        job_data['steps']['segmentation']['status'] = 'success'
        JOBS[job_id]['regions'] = regions
        job_data['steps']['segmentation']['images'] = [
            {'label': 'Dilation Mask', 'src': cv2_to_base64(dilation_vis)},
            {'label': 'Segmented Boxes', 'src': cv2_to_base64(boxes_vis)}
        ]

        # --- STEP 4: Inference ---
        job_data['steps']['modelInference']['status'] = 'processing'
        latex_lines = scanner.image_to_latex(regions)
        job_data['steps']['modelInference']['status'] = 'success'
        JOBS[job_id]['latex_lines'] = latex_lines
        
        # --- STEP 5: Reassembly ---
        job_data['steps']['reassembly']['status'] = 'processing'
        final_doc = scanner.reassemble_document(latex_lines)
        job_data['steps']['reassembly']['status'] = 'success'

        # --- Finalize ---
        job_data['steps']['validationOutput']['status'] = 'success'
        job_data['result'] = final_doc
        job_data['status'] = 'complete'

    except Exception as e:
        print(f"Processing failed: {e}")
        job_data['status'] = 'failed'
        job_data['error'] = str(e)

# --- Global State & Config ---
app = Flask(__name__)
CORS(app) 
scanner = MathScanner(use_model=True) 
JOBS = {} 

TEMP_PROCESSING_DIR = os.path.join(tempfile.gettempdir(), "mathscanner_jobs")
if not os.path.exists(TEMP_PROCESSING_DIR):
    os.makedirs(TEMP_PROCESSING_DIR)

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
        
        JOBS[job_id] = {
            'status': 'uploaded',
            'temp_path': temp_path,
            'result': None,
            'regions': None,
            'latex_lines': None,
            'steps': { 
                'preprocessing': {'status': 'pending', 'images': []},
                'ocrRecognition': {'status': 'pending', 'images': []},
                'segmentation': {'status': 'pending', 'images': []},
                'modelInference': {'status': 'pending', 'images': []},
                'reassembly': {'status': 'pending', 'images': []},
                'validationOutput': {'status': 'pending', 'images': []},
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
    
    if not scanner:
        job_data['status'] = 'failed'
        return jsonify({'success': False, 'error_message': 'MathScanner model failed to load.'}), 503

    # Set status to processing immediately
    job_data['status'] = 'processing'
    thread = threading.Thread(target=run_pipeline_background, args=(job_id,))
    thread.daemon = True
    thread.start()
        
    # Return immediately so the frontend can start polling
    return jsonify({'success': True, 'message': 'Processing started in background'}), 202

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in JOBS:
        return jsonify({'status': 'not_found'}), 404
    status_data = JOBS[job_id].copy()
    status_data.pop('regions', None)
    status_data.pop('latex_lines', None)
    status_data.pop('temp_path', None)
    return jsonify(status_data)

# NEW API ENDPOINT FOR RERUN
@app.route('/api/rerun_equation/<job_id>/<int:index>', methods=['POST'])
def rerun_equation(job_id, index):
    if job_id not in JOBS:
        return jsonify({'success': False, 'error_message': 'Job ID not found'}), 404

    job_data = JOBS[job_id]

    if job_data.get('regions') is None or job_data.get('latex_lines') is None:
        return jsonify({
            'success': False,
            'error_message': 'Segmentation data not available. Must run full pipeline first.'
        }), 400

    regions = job_data['regions']
    latex_lines = job_data['latex_lines']

    if index < 0 or index >= len(regions):
        return jsonify({
            'success': False,
            'error_message': f'Invalid equation index: {index}. Range is 0 to {len(regions) - 1}'
        }), 400

    try:
        # 1. Get the specific region image (numpy array)
        region_to_rerun = regions[index]

        # 2. Run the single inference
        new_latex = scanner.rerun_single_inference(region_to_rerun)

        # 3. Update stored latex lines
        latex_lines[index] = new_latex
        job_data['latex_lines'] = latex_lines  # Reassign

        # 4. Reassemble the full document
        final_doc = scanner.reassemble_document(latex_lines)
        job_data['result'] = final_doc

        # 5. Return the new full output
        return jsonify({
            'success': True,
            'new_latex_output': final_doc
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error_message': f'Rerun failed: {str(e)}'
        }), 500



if __name__ == '__main__':
    app.run(debug=True, port=5000)