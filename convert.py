import shutil
import cv2
import numpy as np
from PIL import Image
import os
import sys
import math

DEBUG_OUTPUT_DIR = "debug_output"

class MathScanner:
    def __init__(self, use_model=True):
        print("--- MathScanner Initialization ---")
        self.use_model = use_model
        self.model = None
        
        if not os.path.exists(DEBUG_OUTPUT_DIR):
            os.makedirs(DEBUG_OUTPUT_DIR)
            print(f"Created debug output directory: '{DEBUG_OUTPUT_DIR}'")
        
        if self.use_model:
            try:
                from pix2tex.cli import LatexOCR
                print("Loading LaTeX-OCR model...")
                self.model = LatexOCR()
                print("Model loaded successfully.")
            except ImportError:
                print("Warning: 'pix2tex' not found. pip install pix2tex")
                self.model = None
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

    def _save_debug_image(self, image, step_name):
        # Ensure BGR format for consistency when saving
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image
            
        output_filename = os.path.join(DEBUG_OUTPUT_DIR, f"{step_name}.png")
        cv2.imwrite(output_filename, display_image)
        print(f"[DEBUG] Saved {step_name}")

    def correct_skew(self, image_path):
        print("\n[STEP 1] Straightening Image...")
        img = cv2.imread(image_path) 
        if img is None:
            raise ValueError(f"Image not found at {image_path}!")

        self._save_debug_image(img, "01a_original_loaded")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert and Otsu threshold for clear edge detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # Detect lines to determine skew angle
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        
        angles = []
        correction_angle = 0.0
        
        if lines is not None:
            line_debug_img = img.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                
                # Filter out near-vertical lines, focus on horizontal text flow
                if abs(angle_deg) < 45 or abs(angle_deg) > 135:
                    angles.append(angle_deg)

            self._save_debug_image(line_debug_img, "01d_detected_lines")

        if not angles:
            print(" > No dominant text lines found. Assuming no skew.")
        else:
            median_angle = np.median(angles)
            # Adjust angle if it's detected as vertical
            if median_angle < -45:
                correction_angle = median_angle + 90
            elif median_angle > 45:
                correction_angle = median_angle - 90
            else:
                correction_angle = median_angle
            print(f" > Correcting by {-correction_angle:.2f} degrees.")
        
        # Rotate image around center
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0) 
        
        # Calculate new bounding dimensions to prevent cutting off corners
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        self._save_debug_image(rotated, "01e_straightened")
        return rotated

    def clean_image(self, img):
        print("\n[STEP 2] Cleaning and Binarizing...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Document Isolation: Find the paper boundary to ignore desk background
        _, binary_for_contour = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        document_mask = np.zeros_like(gray, dtype=np.uint8)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(document_mask, [box], 0, 255, -1) 
            self._save_debug_image(document_mask, "02a_document_mask") 
        else:
            # Fallback: Treat whole image as document
            document_mask.fill(255)

        # 2. Adaptive Thresholding to clean text/shadows
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cleaned_img_full = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10
        )
        self._save_debug_image(cleaned_img_full, "02b_cleaned_full")

        # 3. Apply Mask: Keep text inside paper, force background to white
        temp_binary = cv2.bitwise_and(cleaned_img_full, cleaned_img_full, mask=document_mask)
        inverted_document_mask = cv2.bitwise_not(document_mask)
        binary = cv2.bitwise_or(temp_binary, inverted_document_mask)
        
        # Remove small noise specs
        kernel = np.ones((2,2), np.uint8)
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        self._save_debug_image(clean_binary, "02c_cleaned_binary_final")
        return clean_binary, img

    def segment_regions(self, binary_img, original_img):
        print("\n[STEP 3] Segmenting Equations...")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 12)) 
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        self._save_debug_image(dilated, "03a_dilation_mask")

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        
        # Filter noise
        valid_boxes = []
        for c, box in zip(contours, bounding_boxes):
            x, y, w, h = box
            if w > 20 and h > 20: 
                valid_boxes.append(box)

        if not valid_boxes:
            print(" > No valid equation regions found.")
            return []

        # Smart Sorting (Top-to-Bottom, then Left-to-Right)
        # 1. Sort primarily by Y to find rows
        valid_boxes.sort(key=lambda b: b[1])
        
        sorted_boxes = []
        row_threshold = 20 # pixels
        current_row = [valid_boxes[0]]

        for i in range(1, len(valid_boxes)):
            prev_y = current_row[-1][1]
            curr_y = valid_boxes[i][1]
            
            # If current box is roughly on same line as previous
            if abs(curr_y - prev_y) < row_threshold:
                current_row.append(valid_boxes[i])
            else:
                # Row finished: Sort this row by X (Left to Right)
                current_row.sort(key=lambda b: b[0])
                sorted_boxes.extend(current_row)
                current_row = [valid_boxes[i]]
        
        # Append last row
        current_row.sort(key=lambda b: b[0])
        sorted_boxes.extend(current_row)
        
        regions = []
        debug_draw = original_img.copy() 

        for i, box in enumerate(sorted_boxes):
            x, y, w, h = box
            
            # Add padding for clean crop
            pad = 2
            x_p = max(0, x - pad)
            y_p = max(0, y - pad)
            w_p = min(original_img.shape[1] - x_p, w + 2*pad)
            h_p = min(original_img.shape[0] - y_p, h + 2*pad)

            roi = original_img[y_p:y_p+h_p, x_p:x_p+w_p]
            regions.append(roi)
            
            cv2.rectangle(debug_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_draw, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        self._save_debug_image(debug_draw, "03b_segmented_boxes")
        return regions

    def image_to_latex(self, regions):
        print("\n[STEP 4] Running LatexOCR Model...")
        latex_results = []
        
        for i, region in enumerate(regions):
            # Converting to grayscale for the model input.
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # The model works better with anti-aliased (smooth) edges.            
            # Save the clean crop for debugging
            output_filename = os.path.join(DEBUG_OUTPUT_DIR, f"4a_region_{i+1}_cropped.png")
            cv2.imwrite(output_filename, region_gray)
            
            pil_img = Image.fromarray(region_gray)
            
            try:
                result = self.model(pil_img)
                print(f" > Region {i+1}: {result}")
                latex_results.append(result)
            except Exception as e:
                print(f" > Region {i+1} Failed: {e}")
                latex_results.append("\\text{Error}")

        return latex_results
    
    def reassemble_document(self, latex_list):
        print("\n[STEP 5] Generating Final LaTeX Document...")

        header = r"""\documentclass{article}
    \usepackage{amsmath, amssymb}
    \begin{document}
    """
        footer = r"""
    \end{document}
    """
        body = ""
        for eq in latex_list:
            body += f"\\[\n{eq}\n\\]\n\n"

        full_doc = header + body + footer

        with open(os.path.join(DEBUG_OUTPUT_DIR, "final_output.tex"), "w") as f:
            f.write(full_doc)

        print(f" > Saved to '{DEBUG_OUTPUT_DIR}/final_output.tex'")
        return full_doc

    def _clear_debug_output(self):
        """Removes the existing debug_output directory and recreates it."""
        if os.path.exists(DEBUG_OUTPUT_DIR):
            try:
                # Use shutil.rmtree to remove the directory and all its contents
                shutil.rmtree(DEBUG_OUTPUT_DIR)
                print(f"[CLEANUP] Deleted old '{DEBUG_OUTPUT_DIR}' folder.")
            except OSError as e:
                print(f"[ERROR] Could not remove directory: {e}")
        
        # Recreate the directory to ensure it exists for the new run
        os.makedirs(DEBUG_OUTPUT_DIR)
        print(f"[CLEANUP] Recreated '{DEBUG_OUTPUT_DIR}' folder for new run.")
    
    def run_pipeline(self, image_path):
        print("="*40)
        print(f"STARTING PIPELINE FOR: {image_path}")
        print("="*40)
        self._clear_debug_output()

        try:
            straight_img = self.correct_skew(image_path)
            
            # binary is the mask, clean_bg is the BGR image for cropping
            binary, clean_bg = self.clean_image(straight_img) 
            
            regions = self.segment_regions(binary, clean_bg)
            latex_lines = self.image_to_latex(regions)
            self.reassemble_document(latex_lines)
            
            print("\nPipeline Complete. Check 'debug_output' folder.")
            
        except ValueError as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    converter = MathScanner()
    image_filename = 'images/testing2.png'
    
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
        
    converter.run_pipeline(image_filename)