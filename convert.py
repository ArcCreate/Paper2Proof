import math
import cv2
import numpy as np
from PIL import Image
from pix2tex.cli import LatexOCR
import os

class MathPageDigitizer:
    def __init__(self):
        print("[INIT] Loading LatexOCR Model (this may take a moment)...")
        self.model = LatexOCR()
        # Create output directory for debug images
        if not os.path.exists('debug_output'):
            os.makedirs('debug_output')

    def save_debug_image(self, name, image):
        """Helper to save intermediate images"""
        path = f"debug_output/{name}.png"
        cv2.imwrite(path, image)
        print(f"   > Saved debug image: {path}")

    def correct_skew(self, image_path):
        """
        Step 1: Straighten the image using the Hough Line Transform.
        This is robust against noise and irregular borders.
        """
        print("\n[STEP 1] Straightening Image (Hough Transform)...")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found!")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Binarize and Filter Noise
        # Use a high contrast threshold to isolate the dark ink
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Edge Detection
        # Canny is essential for Hough Transform
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # 3. Probabilistic Hough Line Transform
        # Detect lines in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        
        # Calculate angles of detected lines
        angles = []
        if lines is not None:
            
            # --- Debug: Draw detected lines ---
            line_debug_img = img.copy()
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw the line on the debug image
                cv2.line(line_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Calculate the angle of the line
                # atan2 returns angle in radians between -pi and pi
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                
                # Filter out near-vertical lines, only interested in horizontal text lines
                if abs(angle_deg) < 45 or abs(angle_deg) > 135:
                    angles.append(angle_deg)

            self.save_debug_image("01b_detected_lines", line_debug_img)

        # 4. Determine the dominant angle
        if not angles:
            print("   > No dominant text lines found. Assuming no skew.")
            correction_angle = 0.0
        else:
            # Use the median angle to be robust against outliers
            median_angle = np.median(angles)
            
            # Normalize the angle to be within the horizontal range (-45 to 45)
            if median_angle < -45:
                 correction_angle = median_angle + 90
            elif median_angle > 45:
                 correction_angle = median_angle - 90
            else:
                 correction_angle = median_angle
                 
            print(f"   > Median text line angle: {median_angle:.2f} degrees. Correcting by {-correction_angle:.2f} degrees.")
        
        # 5. Rotate the image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        # We rotate by the negative of the detected angle
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0) 
        
        # Adjust canvas size to avoid clipping the rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        self.save_debug_image("01_straightened", rotated)
        return rotated

    def clean_image(self, img):
        """
        Step 2: Binarize and Clean.
        Removes shadows, wood grain, and paper texture.
        """
        print("\n[STEP 2] Cleaning and Binarizing...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive Thresholding is key here. It calculates threshold for small regions
        # ensuring shadows don't become black blobs.
        # Block Size: 21 (looks at local neighborhood), C: 15 (strictness)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 15
        )

        # Remove small noise (dust/dots)
        # Morphological opening (Erosion followed by Dilation)
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        self.save_debug_image("02_cleaned_binary", clean)
        return clean, img

    def segment_regions(self, binary_img, original_img):
        """
        Step 3: Segmentation.
        Groups ink strokes into equation blocks.
        """
        print("\n[STEP 3] Segmenting Equations...")
        
        # Dilation: Smear the text horizontally to connect characters
        # (50, 5) means connect things far apart horizontally, but keep vertical separation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5)) 
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        self.save_debug_image("03_dilation_mask", dilated)

        # Find contours on the dilated mask
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours top-to-bottom
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1]))

        regions = []
        debug_draw = original_img.copy()

        for i, box in enumerate(bounding_boxes):
            x, y, w, h = box
            
            # Filter noise
            if w < 30 or h < 20:
                continue

            # Add padding to the crop
            pad = 10
            x_p = max(0, x - pad)
            y_p = max(0, y - pad)
            w_p = min(original_img.shape[1] - x_p, w + 2*pad)
            h_p = min(original_img.shape[0] - y_p, h + 2*pad)

            # Crop from the CLEAN ORIGINAL image (not the binary one)
            # We want grayscale/color for the model, not pure black/white usually
            roi = original_img[y_p:y_p+h_p, x_p:x_p+w_p]
            
            regions.append(roi)
            
            # Draw box for visualization
            cv2.rectangle(debug_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_draw, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        self.save_debug_image("04_segmented_boxes", debug_draw)
        print(f"   > Found {len(regions)} equation regions.")
        return regions

    def recognize_latex(self, regions):
        """
        Step 4: AI Recognition.
        Passes each cropped image to the Pix2Tex model.
        """
        print("\n[STEP 4] Running LatexOCR Model...")
        latex_results = []
        
        for i, region in enumerate(regions):
            # Convert CV2 array (BGR) to PIL Image (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            
            try:
                # Run Model
                result = self.model(pil_img)
                print(f"   > Region {i+1}: {result}")
                latex_results.append(result)
            except Exception as e:
                print(f"   > Region {i+1} Failed: {e}")
                latex_results.append("% Error reading equation")

        return latex_results

    def reassemble_document(self, latex_list):
        """
        Step 5: Output Generation.
        Creates the final .tex file.
        """
        print("\n[STEP 5] Genering Final LaTeX Document...")
        
        header = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
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
            
        print("   > Saved to 'final_output.tex'")
        return full_doc

    def run_pipeline(self, image_path):
        print("="*40)
        print(f"STARTING PIPELINE FOR: {image_path}")
        print("="*40)

        # 1. Straighten
        straight_img = self.correct_skew(image_path)
        
        # 2. Clean
        binary, clean_bg = self.clean_image(straight_img)
        
        # 3. Segment
        regions = self.segment_regions(binary, clean_bg)
        
        # 4. Recognize
        latex_lines = self.recognize_latex(regions)
        
        # 5. Output
        self.reassemble_document(latex_lines)
        
        print("\nPipeline Complete. Check 'debug_output' folder for images.")

if __name__ == "__main__":
    converter = MathPageDigitizer()
    converter.run_pipeline('images/fullpage.png')