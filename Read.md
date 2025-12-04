# pix2tex Setup and Usage on Attu

## Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```
## Install pix2tex, opencv, matplotlib
```bash
pip install --upgrade pip
pip install "pix2tex[gui]"
pip install opencv-python
pip install matplotlib
```
## Command Line usage of pix2tex
```bash
pix2tex images/my_equation.png
```

## Api usage of pix2tex
```bash
pip install -U "pix2tex[api]"
python -m pix2tex.api.run
```

## using the python file
```bash
python convert.py
```

