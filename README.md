# 🧾 Automated Score Sheet Analysis using OCR and Computer Vision

## 🔍 Overview
This project implements an automated system for analyzing scanned score sheets using a combination of **Computer Vision techniques** and **Optical Character Recognition (OCR)**. The system detects table structures, extracts individual cells, recognizes text and numeric scores, computes totals, and overlays the extracted information directly onto the document.

The project is designed to handle skewed, noisy, and real-world scanned documents efficiently.

---

## 🧠 Problem Statement
Manual evaluation and digitization of score sheets is time-consuming and prone to errors. This project automates the process by detecting table layouts, recognizing scores using OCR, and generating visually verified results.

---

## 🛠️ Technologies Used
- **Python**
- **OpenCV**
- **PaddleOCR**
- **NumPy**
- Computer Vision
- Optical Character Recognition (OCR)

---

## ⚙️ Approach

### 1️⃣ Image Deskewing
Skew in scanned score sheets is corrected using **Hough Line Detection**. The median angle of detected lines is computed and used to rotate the image, ensuring proper alignment of table cells.

### 2️⃣ Table and Cell Detection
Adaptive thresholding and morphological operations are applied to detect horizontal and vertical lines. These lines are combined to form a grid structure, and individual cells are extracted and grouped into rows.

### 3️⃣ Text Recognition using PaddleOCR
Each detected cell undergoes multiple preprocessing strategies, including resizing and adaptive thresholding. **PaddleOCR** is used to recognize both textual and numeric content, with special handling for score rows to improve accuracy.

### 4️⃣ Score Computation and Validation
Numeric values are validated, corrected, and aggregated to compute total scores. Constraints such as maximum score limits are enforced to ensure correctness.

### 5️⃣ Result Visualization and Output
Detected cells are highlighted, recognized content is overlaid for verification, and the final processed image is saved to the results directory.

---

## 📥 Input
- Scanned score sheet images
- Tabular layout with numeric score values

---

## 📤 Output
- Annotated images with detected table cells and recognized scores
- Computed total score overlaid on the document

---

## ▶️ Execution

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Script
```bash
python A2_solution.py
```

## 👤Author
Anum Asghar
