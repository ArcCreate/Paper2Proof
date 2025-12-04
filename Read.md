# pix2tex Setup and Usage on Attu

## Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```
## Install pix2tex
```bash
pip install --upgrade pip
pip install "pix2tex[gui]"
```
## Command Line usage
```bash
pix2tex images/my_equation.png
```
## IPython usage
```bash
python convert.py
```

## Api usage
```bash
pip install -U "pix2tex[api]"
python -m pix2tex.api.run

```
