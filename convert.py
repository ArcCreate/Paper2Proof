import math
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Conditional import for LatexOCR to handle environments where pix2tex might not be installed
try:
    from pix2tex.cli import LatexOCR
    _LATEX_OCR_AVAILABLE = True
except ImportError:
    _LATEX_OCR_AVAILABLE = False
    print("Warning: 'pix2tex' library not found. LaTeX generation will be skipped.")
    print("To install: pip install pix2tex")
except Exception as e:
    _LATEX_OCR_AVAILABLE = False
    print(f"Error loading LatexOCR: {e}")

class MathPageDigitizer:
    """
    A comprehensive class for digitizing a full page of handwritten or printed math 
    into a structured LaTeX document. Includes robust image cleaning, deskewing, 
    equation segmentation, and debug image output.
    """
    def __init__(self):
        self.model = None
        # Create output directory for debug images
        self.debug_dir = 'debug_output'
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        if _LATEX_OCR_AVAILABLE:
            print("[INIT] Loading LatexOCR Model (this may take a moment)...")
            try:
                self.model = LatexOCR() 
                print("[INIT] Model loaded successfully.")
            except Exception as e:
                print(f"[INIT] Error loading model: {e}")
                self.model = None
        else:
            print("[INIT] LatexOCR Model not loaded.")


    def save_debug_image(self, name, image):
        """Helper to save intermediate images for debugging and pipeline inspection."""
        path = f"{self.debug_dir}/{name}.png"
        cv2.imwrite(path, image)
        print(f"   > Saved debug image: {path}")

    ## --- STEP 1: DESKEW ---
    def correct_skew(self, image_path):
        """
        Straightens the image using the Hough Line Transform to find the dominant text angle.
        """
        print("\n[STEP 1] Straightening Image (Hough Transform)...")
        img = cv2.imread(image_path) 
        if img is None:
            raise ValueError(f"Image not found at {image_path}!")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Binarize and Filter Noise
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Edge Detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # 3. Probabilistic Hough Line Transform
        # Detect lines based on edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        
        angles = []
        if lines is not None:
            line_debug_img = img.copy()
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw detected lines on a debug copy
                cv2.line(line_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Calculate angle in degrees
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                
                # Only consider near-horizontal lines
                if abs(angle_deg) < 45 or abs(angle_deg) > 135:
                    angles.append(angle_deg)

            self.save_debug_image("01b_detected_lines", line_debug_img)

        # 4. Determine the dominant angle
        if not angles:
            print("   > No dominant text lines found. Assuming no skew.")
            correction_angle = 0.0
        else:
            median_angle = np.median(angles)
            
            # Normalize angle to the nearest 0 or 90
            if median_angle < -45:
                correction_angle = median_angle + 90
            elif median_angle > 45:
                correction_angle = median_angle - 90
            else:
                correction_angle = median_angle
                    
            print(f"   > Median text line angle: {median_angle:.2f} degrees. Correcting by {-correction_angle:.2f} degrees.")
        
        # 5. Rotate the image (with canvas size adjustment)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0) 
        
        # Calculate new dimensions for the rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the translation components of the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform the affine warp with white border
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        self.save_debug_image("01_straightened", rotated)
        return rotated

    ## --- STEP 2: CLEANING & BINARIZATION ---
    def clean_image(self, img):
        """
        Binarizes and Cleans the image by finding the paper boundary and applying adaptive 
        thresholding only within that area, forcing the outside background to pure white.
        """
        print("\n[STEP 2] Cleaning and Binarizing (Document Isolation)...")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 1. Document Isolation: Find the paper boundary ---
        # Otsu's thresholding separates the light paper (255) from the dark background/text (0).
        _, binary_for_contour = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        document_mask = np.full_like(gray, 255, dtype=np.uint8) 
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            document_mask = np.zeros_like(gray, dtype=np.uint8)
            # Fill the paper area with white (255) based on the boundary box
            cv2.drawContours(document_mask, [box], 0, 255, -1) 
            self.save_debug_image("02a_document_mask", document_mask) 
        else:
            print("   > WARNING: Could not detect paper boundary. Cleaning entire image.")

        # --- 2. Enhanced Adaptive Thresholding on the full image ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Threshold (Text: 0, Everything else: 255)
        cleaned_img_full = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10 # Block size 25, C value 10
        )
        self.save_debug_image("02b_cleaned_full_adaptive_result", cleaned_img_full)

        # --- 3. Combine: Keep cleaned text only inside the paper mask ---
        
        # a. Apply the document mask: Cleans the text on paper, but turns the background BLACK (0)
        temp_binary = cv2.bitwise_and(cleaned_img_full, cleaned_img_full, mask=document_mask)

        # b. Invert the document mask: Paper is Black (0). Background is White (255).
        inverted_document_mask = cv2.bitwise_not(document_mask)
        
        # c. Combine: Use OR to force the background to white (255), keeping the text intact
        binary = cv2.bitwise_or(temp_binary, inverted_document_mask)
        
        # --- 4. Morphological Cleaning ---
        # Remove small noise (dust/dots)
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        self.save_debug_image("02_cleaned_binary", clean)
        # Return the final clean binary image AND the straightened color image (for cropping)
        return clean, img

    ## --- STEP 3: SEGMENTATION ---
    def segment_regions(self, binary_img, original_img):
        """
        Segments the single-page binary image into individual equation regions 
        by using a wide, short dilation kernel.
        """
        print("\n[STEP 3] Segmenting Equations (Refined Dilation)...")
        
        # Dilation: Wide (100) to connect horizontal parts, short (3) to prevent vertical connection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3)) 
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        self.save_debug_image("03_dilation_mask_refined", dilated)

        # Find contours on the dilated mask
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = []
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out extremely small noise contours
            if w > 40 and h > 25: 
                valid_boxes.append((x, y, w, h))

        # Sort the bounding boxes top-to-bottom (based on y-coordinate)
        valid_boxes.sort(key=lambda b: b[1])
        
        regions = []
        debug_draw = original_img.copy() 

        for i, box in enumerate(valid_boxes):
            x, y, w, h = box
            
            # Add padding to the crop
            pad = 10
            x_p = max(0, x - pad)
            y_p = max(0, y - pad)
            
            # Recalculate width/height considering the boundary limits
            w_p = min(original_img.shape[1] - x_p, w + 2*pad)
            h_p = min(original_img.shape[0] - y_p, h + 2*pad)

            # Crop from the straightened color image (original_img is 'clean_bg')
            roi = original_img[y_p:y_p+h_p, x_p:x_p+w_p]
            
            regions.append(roi)
            
            # Draw box and index for visualization
            cv2.rectangle(debug_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_draw, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        self.save_debug_image("04_segmented_boxes_refined", debug_draw)
        print(f"   > Found {len(regions)} equation regions.")
        return regions

    ## --- STEP 4: RECOGNITION ---
    def recognize_latex(self, regions):
        """
        Passes each cropped image to the LatexOCR model.
        """
        if self.model is None:
            return ["% LaTeX generation skipped (Model not loaded)."] * len(regions)
        
        print("\n[STEP 4] Running LatexOCR Model...")
        latex_results = []
        
        for i, region in enumerate(regions):
            # Convert CV2 array (BGR) to PIL Image (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            
            try:
                # Run Model
                result = self.model(pil_img)
                print(f"   > Region {i+1}: {result}")
                latex_results.append(result)
            except Exception as e:
                print(f"   > Region {i+1} Failed: {e}")
                latex_results.append("% Error reading equation")

        return latex_results

    ## --- STEP 5: OUTPUT ---
    def reassemble_document(self, latex_list):
        """
        Creates the final .tex file with a minimal, clean preamble.
        """
        print("\n[STEP 5] Genering Final LaTeX Document...")
        
        # Minimal and robust LaTeX preamble
        header = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fontspec} 
\usepackage[english]{babel} 
\babelfont{rm}{Noto Sans} 
\begin{document}
"""
        footer = r"""
\end{document}
"""
        body = ""
        for eq in latex_list:
            # Wrap each equation in a display math block
            body += f"$$ {eq} $$\n\n"

        full_doc = header + body + footer
        
        with open("final_output.tex", "w") as f:
            f.write(full_doc)
            
        print("   > Saved final LaTeX to 'final_output.tex'")
        return full_doc
        
    def run_pipeline(self, image_path):
        print("="*40)
        print(f"STARTING PIPELINE FOR: {image_path}")
        print("="*40)

        try:
            # 1. Straighten
            straight_img = self.correct_skew(image_path)
            
            # 2. Clean & Binarize (binary: the black-and-white text mask, straight_img is the color background)
            binary, clean_bg = self.clean_image(straight_img)
            
            # 3. Segment (uses the binary mask to find regions, but crops from the clean_bg color image)
            regions = self.segment_regions(binary, clean_bg)
            
            # 4. Recognize
            latex_lines = self.recognize_latex(regions)
            
            # 5. Output
            self.reassemble_document(latex_lines)
            
            print("\nPipeline Complete. Check 'debug_output' folder for images.")
            
        except ValueError as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            print("Please ensure the image file path is correct.")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    converter = MathPageDigitizer()
    
    # Use a default filename or take one from command line arguments
    image_filename = 'images/img1.png' 
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
    
    converter.run_pipeline(image_filename)