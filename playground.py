import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def process_document_image(image_path):
    """
    Loads an image and applies cleaning, shadow removal, and basic deskewing
    to enhance the handwritten text.
    """
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale, as most image processing is done on single channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. **Cleaning and Shadow/Texture Removal (The Key Step)**
    # Apply Gaussian blur to reduce high-frequency noise and paper texture.
    # This smooths the image, making the shadows and text distinction clearer.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding is crucial for images with uneven lighting (shadows).
    # It calculates a different threshold for every small region of the image, 
    # allowing it to remove shadows and background texture successfully.
    # - 255: Max value for output (white)
    # - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Uses a Gaussian-weighted sum of the neighborhood
    # - cv2.THRESH_BINARY_INV: Inverts the result (black text on white background)
    # - 25: Block Size (must be odd) - the area used to calculate the threshold
    # - 10: C - Constant subtracted from the mean to fine-tune sensitivity
    cleaned_img = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        25, 
        10
    )

    # 3. **Simple Straightening (Deskewing) using Contours**
    # This attempts to fix minor rotational tilt by finding the overall text contour.
    
    # Find contours on the cleaned image
    contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, which is usually the bulk of the text
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Adjust the angle to correct for rotation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate the image using the calculated angle
        (h, w) = cleaned_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_img = cv2.warpAffine(cleaned_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed_img = cleaned_img # Fallback
    
    # 4. Save and Display Results
    output_path = "processed_" + os.path.basename(image_path)
    cv2.imwrite(output_path, deskewed_img)

    print(f"Processed image saved to: {output_path}")

    # Visualization using Matplotlib (optional for local running)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # We display the inverse of the binary image to match the request (white text on black background for some systems, but here, we keep it black text on white background)
    axes[1].imshow(cv2.bitwise_not(deskewed_img), cmap='gray')
    axes[1].set_title('Processed Image (Cleaned, Enhanced, Deskewed)')
    axes[1].axis('off')

    plt.suptitle("Document Image Processing", fontsize=16)
    plt.show()

# --- Example Usage ---

if __name__ == '__main__':
    # NOTE: The 'uploaded:' path is only available in this environment. 
    # To run this code locally, replace this with the actual path to your saved image file, e.g., "my_notes.png"
    input_file = "debug_output/01_straightened.png" 
    
    try:
        process_document_image(input_file)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("\n--- Setup Instructions ---")
        print("1. Install libraries: pip install opencv-python numpy matplotlib")
        print("2. Ensure the input image file path is correct.")