# pix2tex Setup and Usage on Attu

### Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### Install python dependencies for backend
```bash
pip install --upgrade pip
# Install pix2tex with the GUI/API extras
pip install "pix2tex[gui]" 
# Install Flask, CORS, OpenCV, NumPy, etc.
pip install Flask Flask-Cors opencv-python numpy Pillow werkzeug
```

### Frontend react dependencies
```bash
npm install
```
# How to run the Application

### In Terminal 1 start the backend server
```bash
python app.py
```

### In Terminal 2 start the frontend React server
```bash
npm start
```

# Direct usage for testing/debugging
```bash
pix2tex images/my_equation.png

# Run the pipeline with the default test image:
python convert.py

# Run the pipeline with a custom image:
python convert.py /path/to/your/image.jpg
```

