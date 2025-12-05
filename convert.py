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

    def _merge_boxes(self, boxes, overlap_threshold=0.3):
        """
        Merges overlapping bounding boxes using IoU-like logic to consolidate 
        fragmented equation parts.
        """
        if not boxes:
            return []
            
        # Convert to numpy array for easier manipulation
        boxes_np = np.array(boxes)
        
        # Calculate area
        area = (boxes_np[:, 2]) * (boxes_np[:, 3]) # w * h

        # Initialize list to keep track of boxes to be kept
        merged_boxes = []
        
        # A list to keep track of which boxes have already been included in a merged box
        is_merged = [False] * len(boxes_np)

        for i in range(len(boxes_np)):
            if is_merged[i]:
                continue

            current_box = boxes_np[i]
            x1, y1, w1, h1 = current_box
            x2, y2 = x1 + w1, y1 + h1
            
            # Start a new merged box with the current box
            merged_rect = current_box.copy()
            is_merged[i] = True
            
            # Iterate through the rest of the boxes
            for j in range(i + 1, len(boxes_np)):
                if is_merged[j]:
                    continue
                    
                next_box = boxes_np[j]
                nx1, ny1, nw1, nh1 = next_box
                nx2, ny2 = nx1 + nw1, ny1 + nh1
                
                # Calculate Intersection coordinates
                inter_x1 = max(x1, nx1)
                inter_y1 = max(y1, ny1)
                inter_x2 = min(x2, nx2)
                inter_y2 = min(y2, ny2)
                
                # Calculate Intersection area
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                intersection_area = inter_w * inter_h
                
                # Check for significant overlap with either box
                area_ratio_1 = intersection_area / area[i]
                area_ratio_2 = intersection_area / area[j]
                
                if area_ratio_1 > overlap_threshold or area_ratio_2 > overlap_threshold:
                    # Merge boxes (calculate the new minimal enclosing bounding box)
                    new_x1 = min(merged_rect[0], nx1)
                    new_y1 = min(merged_rect[1], ny1)
                    new_x2 = max(merged_rect[0] + merged_rect[2], nx2)
                    new_y2 = max(merged_rect[1] + merged_rect[3], ny2)
                    
                    merged_rect = np.array([new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1])
                    is_merged[j] = True
            
            merged_boxes.append(merged_rect.tolist())
            
        return merged_boxes

    def segment_regions(self, binary_img, original_img):
        print("\n[STEP 3] Segmenting Equations...")
        
        # --- 1. Initial Contour Detection ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 12)) 
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        self._save_debug_image(dilated, "03a_dilation_mask")

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        
        # Filter noise (assuming a small minimum size for equation regions)
        valid_boxes = []
        for c, box in zip(contours, bounding_boxes):
            x, y, w, h = box
            if w > 20 and h > 20: 
                valid_boxes.append(box)

        if not valid_boxes:
            print(" > No valid equation regions found after initial filter.")
            # FIX: Return two values even on failure
            return [], original_img 
            
        # --- 2. Bounding Box Merging ---
        print(f" > Found {len(valid_boxes)} initial regions. Merging overlaps...")
        # Use a slightly stricter overlap threshold for initial grouping
        merged_boxes = self._merge_boxes(valid_boxes, overlap_threshold=0.25)
        print(f" > Consolidated to {len(merged_boxes)} regions after merging.")

        # --- 3. Smart Sorting (Top-to-Bottom, then Left-to-Right) ---
        merged_boxes.sort(key=lambda b: b[1])
        
        sorted_boxes = []
        current_row = []
        
        all_heights = [b[3] for b in merged_boxes]
        median_height = np.median(all_heights) if all_heights else 30 
        row_threshold = median_height * 0.7 
        
        for box in merged_boxes:
            if not current_row:
                current_row.append(box)
                continue

            prev_box_y = np.mean([b[1] + b[3]/2 for b in current_row])
            curr_box_y_center = box[1] + box[3]/2
            
            if abs(curr_box_y_center - prev_box_y) < row_threshold:
                current_row.append(box)
            else:
                # Row finished: Sort this row by X (Left to Right)
                current_row.sort(key=lambda b: b[0])
                sorted_boxes.extend(current_row)
                current_row = [box] # Start new row
        
        # Append last row
        if current_row:
            current_row.sort(key=lambda b: b[0])
            sorted_boxes.extend(current_row)
        
        # --- 4. Crop Regions and Draw Debug Image ---
        regions = []
        debug_draw = original_img.copy() 

        for i, box in enumerate(sorted_boxes):
            x, y, w, h = box
            
            # Add padding for clean crop (3 pixels)
            pad = 3
            x_p = max(0, x - pad)
            y_p = max(0, y - pad)
            # Ensure width/height do not exceed image boundaries
            w_p = min(original_img.shape[1] - x_p, w + 2*pad)
            h_p = min(original_img.shape[0] - y_p, h + 2*pad)

            roi = original_img[y_p:y_p+h_p, x_p:x_p+w_p]
            regions.append(roi)
            
            # Draw bounding box and index on the debug image
            cv2.rectangle(debug_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_draw, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        self._save_debug_image(debug_draw, "03b_segmented_boxes_refined")
        
        # FIX APPLIED: Always return two values: regions and the debug image.
        return regions, debug_draw
    
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
        # We comment out _clear_debug_output() for safety in a web environment, 
        # as multiple users might run it concurrently, but it's fine for initial testing.
        # self._clear_debug_output() 

        try:
            straight_img = self.correct_skew(image_path)
            
            binary, clean_bg = self.clean_image(straight_img) 
            
            regions = self.segment_regions(binary, clean_bg)
            latex_lines = self.image_to_latex(regions)
            final_latex_doc = self.reassemble_document(latex_lines) # Capture the result
            
            print("\nPipeline Complete. Check 'debug_output' folder.")
            
            # <<< IMPORTANT CHANGE: Return the result >>>
            return final_latex_doc
            
        except ValueError as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            raise  # Re-raise the error to be caught by the Flask route
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            raise # Re-raise the error

if __name__ == "__main__":
    converter = MathScanner()
    image_filename = 'images/testing1.png'
    
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
        
    converter.run_pipeline(image_filename)