import cv2
import numpy as np
from PIL import Image
import os
import sys
import math # <--- New import required for the deskew logic

# Define the debug output folder
DEBUG_OUTPUT_DIR = "debug_output"

class MathScanner:
    def __init__(self, use_model=True):
        print("--- MathScanner Initialization ---")
        self.use_model = use_model
        self.model = None
        
        # 1. Setup Debug Folder
        if not os.path.exists(DEBUG_OUTPUT_DIR):
            os.makedirs(DEBUG_OUTPUT_DIR)
            print(f"Created debug output directory: '{DEBUG_OUTPUT_DIR}'")
        
        # 2. Try to load the Math-to-Latex model
        if self.use_model:
            try:
                # Assuming 'pix2tex' is installed in the environment
                from pix2tex.cli import LatexOCR
                print("Loading LaTeX-OCR model... (this may take a moment)")
                self.model = LatexOCR()
                print("Model loaded successfully.")
            except ImportError:
                print("Warning: 'pix2tex' library not found. LaTeX generation will be skipped.")
                print("To install: pip install pix2tex")
                self.model = None
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

    def _save_debug_image(self, image, step_name):
        """Helper function to save intermediate images to the debug folder."""
        # Convert grayscale images back to BGR before saving if needed (for consistency)
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image
            
        output_filename = os.path.join(DEBUG_OUTPUT_DIR, f"{step_name}.png")
        cv2.imwrite(output_filename, display_image)
        print(f"[DEBUG] Saved {step_name} image to: {output_filename}")
        # 

    def process_image(self, image_path, output_path="cleaned_equation_final.png"):
        """
        Main pipeline: Deskew -> Shadow Removal -> Save -> LaTeX
        NOTE: Image loading moved inside deskew for path-based logic.
        """
        print(f"\n--- Starting Image Processing Pipeline ---")
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} not found.")
            return

        print(f"Processing image file: {image_path}")
        
        # 1. Deskew (Straighten) - **Uses path directly**
        straightened = self.correct_skew(image_path)
        
        # 2. Clean (Shadow Removal & Binarization) - **Uses array**
        print("2. Starting Shadow Removal and Binarization process...")
        cleaned = self.remove_shadows_and_binarize(straightened)
        self._save_debug_image(cleaned, "02_shadows_removed_binarized")

        # 3. Save the final processed image
        final_output_path = os.path.join(DEBUG_OUTPUT_DIR, output_path)
        cv2.imwrite(final_output_path, cleaned)
        print(f"3. Final cleaned image saved to: {final_output_path}")

        # 4. Generate LaTeX
        print("4. Starting LaTeX Generation...")
        latex = self.image_to_latex(cleaned)
        
        return latex

## --- STEP 1: DESKEW (New Logic) ---
    def correct_skew(self, image_path):
        """
        Straightens the image using the Hough Line Transform to find the dominant text angle.
        """
        print("\n[STEP 1] Straightening Image (Hough Transform)...")
        img = cv2.imread(image_path) 
        if img is None:
            raise ValueError(f"Image not found at {image_path}!")

        self._save_debug_image(img, "01a_original_loaded")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Binarize and Filter Noise
        print(" > Binarizing image for edge detection.")
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._save_debug_image(thresh, "01b_deskew_threshold")
        
        # 2. Edge Detection
        print(" > Applying Canny edge detection.")
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        self._save_debug_image(edges, "01c_canny_edges")
        
        # 3. Probabilistic Hough Line Transform
        print(" > Detecting lines using Hough Line Transform.")
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        
        angles = []
        correction_angle = 0.0
        
        if lines is not None:
            line_debug_img = img.copy()
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw detected lines on a debug copy
                cv2.line(line_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Calculate angle in degrees
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                
                # Only consider lines near horizontal or vertical (e.g., text baselines)
                if abs(angle_deg) < 45 or abs(angle_deg) > 135:
                    angles.append(angle_deg)
                
            self._save_debug_image(line_debug_img, "01d_detected_lines_on_original")

        # 4. Determine the dominant angle
        if not angles:
            print(" > No dominant text lines found. Assuming no skew (0.0 degrees).")
        else:
            median_angle = np.median(angles)
            
            # Normalize angle to the nearest 0 or 90
            if median_angle < -45:
                # e.g., -88 degrees becomes -88 + 90 = 2 degrees
                correction_angle = median_angle + 90
            elif median_angle > 45:
                # e.g., 88 degrees becomes 88 - 90 = -2 degrees
                correction_angle = median_angle - 90
            else:
                # e.g., -5 degrees stays -5 degrees
                correction_angle = median_angle
                    
            print(f" > Median text line angle: {median_angle:.2f} degrees. Correction angle: {-correction_angle:.2f} degrees.")
        
        # 5. Rotate the image (with canvas size adjustment)
        print(f" > Rotating image by {-correction_angle:.2f} degrees.")
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0) 
        
        # Calculate new dimensions for the rotated image to prevent cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the translation components of the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform the affine warp with white border
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        self._save_debug_image(rotated, "01e_straightened")
        print("[STEP 1] Deskewing complete.")
        return rotated

    def remove_shadows_and_binarize(self, image):
        """
        Removes shadows using background division and applies final thresholding.
        (Logic remains unchanged from previous version)
        """
        print("[CLEAN] Splitting image into RGB channels.")
        planes = cv2.split(image)
        result_planes = []
        
        for i, plane in enumerate(planes):
            # 1. Dilate the image to remove text, leaving only the background/shadows
            print(f"[CLEAN] Channel {i}: Dilating to estimate background.")
            dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
            
            # 2. Median blur to smooth the background estimate
            print(f"[CLEAN] Channel {i}: Applying median blur to background.")
            bg_img = cv2.medianBlur(dilated, 21)
            
            # Save the estimated background for debugging
            if i == 0:
                 self._save_debug_image(bg_img, "02a_estimated_background_channel0")

            # 3. Calculate difference: |255 - (Background - Original)|
            print(f"[CLEAN] Channel {i}: Calculating difference (shadow removal).")
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            
            # 4. Normalize to use full dynamic range
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
            result_planes.append(norm_img)
            
        # Merge back to color (though it looks gray)
        print("[CLEAN] Merging normalized channels.")
        result_normalized = cv2.merge(result_planes)
        self._save_debug_image(result_normalized, "02b_normalized_result_merged")
        
        # Convert to grayscale for final thresholding
        gray = cv2.cvtColor(result_normalized, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding for crisp black and white text
        print("[CLEAN] Applying final Otsu's thresholding.")
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        print("[CLEAN] Image cleaning complete.")
        return thresh

    def image_to_latex(self, img_array):
        """
        Feeds the clean image into the LaTeX-OCR model.
        (Logic remains unchanged from previous version)
        """
        print("[LATEX] Starting LaTeX generation.")
        if self.model is None:
            print("[LATEX] Model not loaded. Skipping LaTeX generation.")
            return "LaTeX generation skipped (Model not loaded)."
            
        # Convert OpenCV array (numpy) to PIL Image for the model
        print("[LATEX] Converting numpy array to PIL Image.")
        pil_image = Image.fromarray(img_array)
        
        print("[LATEX] Calling model for prediction...")
        latex_code = self.model(pil_image)
        print("[LATEX] Prediction complete.")
        return latex_code

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- PROGRAM START ---")
    
    # Create the scanner
    scanner = MathScanner()
    
    # Example usage:
    image_filename = 'images/fullpage.png' 
    
    # If specific file is passed as argument
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]

    # Run pipeline
    latex_output = scanner.process_image(image_filename, output_path="cleaned_equation_final.png")
    
    print("\n--- Final LaTeX Output ---\n")
    print(latex_output)
    print("\n--------------------------\n")