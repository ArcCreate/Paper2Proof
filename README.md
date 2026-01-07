# Paper2Proof: Offline Full-Page LaTeX Conversion
---

## 📄 Downloads


- **[Download Presentation Slides](Paper.pdf)**
- **[Download Paper (PDF)](Paper2Proof.pdf)**

---

## Overview

Paper2Proof is an **offline pipeline for converting full pages of handwritten or printed mathematics into LaTeX**. Rather than relying on large, online multimodal models, this project extends the capabilities of the lightweight, open-source **pix2tex (LaTeX-OCR)** model by pairing it with a classical computer vision segmentation pipeline.

Our key insight is that **page-level LaTeX conversion can be reduced to single-equation OCR** by reliably segmenting individual equations from a document. This allows existing single-formula OCR models to scale to full pages **without retraining**.

---

## Motivation

Typing LaTeX—especially for complex mathematics—is time-consuming and error-prone. While modern OCR systems perform well on single equations, they typically fail on full-page documents or require heavy online models.

Paper2Proof bridges this gap by:
- Operating fully **offline**
- Using **classical computer vision** for page understanding
- Leveraging **pix2tex** for equation-level recognition

---

## Method

The pipeline consists of five main stages:

1. **Skew Correction**
   - Edge detection + Hough line transform
   - Global rotation using median line angle

2. **Image Cleaning & Binarization**
   - Gaussian blur
   - Adaptive thresholding
   - Morphological opening to remove noise

3. **Equation Segmentation**
   - Horizontal dilation to connect equation characters
   - Connected-component analysis
   - Bounding box merging for superscripts and accents

4. **Equation Recognition**
   - Apply pix2tex independently to each detected equation

5. **Document Reconstruction**
   - Combine LaTeX outputs
   - Wrap in valid LaTeX environments

This design favors **high recall**, ensuring equations are not missed, even at the cost of some false positives.

---

## Results

### Segmentation Performance
- **Recall:** 1.00  
- **Precision:** 0.70  
- **Segmentation Rate:** 0.82  

### End-to-End Performance
- **Normalized Edit Distance:** 0.24  
- **Exact Match Rate:** 0.15  

When segmentation succeeds, recognition quality closely matches pix2tex’s original single-equation performance. Errors are typically minor and easy to correct manually.

---

## Key Limitations

- Sensitivity to uneven lighting and handwritten artifacts
- Difficulty handling equations on the same row (arrays)
- OCR confusion between visually similar symbols (e.g., Φ vs φ)
- Weak recognition of digits due to training data bias

1. Harkness, I., & Watson, J. *Improving efficiency and consistency of student learning assessments: A new framework using LaTeX.* ASEE 2024.  
2. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
