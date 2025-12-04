from PIL import Image
from pix2tex.cli import LatexOCR

# Load the image
img = Image.open('images/fullpage.png')

# Initialize the LaTeX OCR model
model = LatexOCR()

# Convert image to LaTeX
latex_code = model(img)

print(latex_code)