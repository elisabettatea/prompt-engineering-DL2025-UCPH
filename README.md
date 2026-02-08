# Lung Segmentation using Segment Anything Model (SAM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![SAM](https://img.shields.io/badge/Meta%20AI-Segment%20Anything-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This project implements and evaluates **Prompt Engineering** strategies to adapt Meta AI's **Segment Anything Model (SAM)** for automatic lung segmentation in chest X-ray images.

Developed as part of the *Advanced Deep Learning (2025)* course at the University of Copenhagen (UCPH), this repository demonstrates how foundation models can be adapted via prompting (zero-shot / few-shot transfer) to handle domain-specific medical data without full retraining.

## 🚀 Key Features

- **Data Preprocessing Pipeline**  
  Transformation of grayscale medical X-rays from float `[0,1]` to SAM-compatible RGB `uint8 [0,255]`.

- **Prompt Engineering**  
  Implementation and comparison of three different prompting strategies:
  1. Sparse point prompts (foreground/background)
  2. Bounding box prompts
  3. Hybrid prompts (multi-point with refinement)

- **Quantitative Evaluation**  
  Automated evaluation using F1-score on a validation dataset.

- **Qualitative Analysis**  
  Visual inspection of predicted segmentation masks compared to ground truth.

---

## 📊 Methodology

### 1. Data Loading & Preprocessing

The original dataset consists of grayscale chest X-ray images with shape `(N, H, W, 1)` and floating-point values in `[0,1]`.

Since SAM expects RGB images, the following preprocessing steps were applied:

- Scale pixel values from `[0,1]` to `[0,255]` and cast to `uint8`
- Replicate the single channel three times to obtain `(N, H, W, 3)`
- Squeeze ground-truth masks to binary format

---

### 2. Prompting Strategies

We investigated how different spatial prompts influence SAM’s lung segmentation performance.

#### Approach 1: Single Foreground + Background Point

A minimal setup using one foreground point inside the lung and one background point on the spinal column.

```python
input_points = np.array([
    [100, 128],  # foreground (lung)
    [128, 128]   # background (spinal column)
])
input_labels = np.array([1, 0])
```

### Observation

Poor performance. Increasing SAM’s internal confidence scores did not improve segmentation quality relative to the ground truth.

---

## Approach 2: Bounding Box Prompts (Selected Method)

Explicit localization using rectangular bounding boxes for the left and right lungs based on anatomical layout.

```python
input_boxes = torch.tensor([
    [35, 30, 115, 220],   # left lung
    [155, 30, 235, 220], # right lung
], dtype=torch.float32, device=predictor.device)
```

### Observation

Best overall performance. The masks generated from the two boxes are combined using a logical OR operation, resulting in accurate lung boundaries.

---

## Approach 3: Multi-Point Prompting (FG + BG)

Two foreground points (one per lung) and one background point on the spine, using SAM’s mask refinement.

### Observation

Better than the single-point approach, but less precise than bounding boxes, with some leakage into surrounding anatomical structures.

---

## 📊 Qualitative Results

### Approach 1: Single Foreground + Background Point
Minimal point-based prompt. SAM struggles with this approach; masks are incomplete and low quality.

![Single Point 1](plot%20results/segm1-1.png)
![Single Point 2](plot%20results/segm1-2.png)
![Single Point 3](plot%20results/segm1-3.png)

---

### Approach 2: Bounding Box Prompts (Selected)
Bounding boxes for left and right lungs. Masks combined with logical OR. Provides the most accurate segmentation.

![Bounding Boxes](plot%20results/segm2.png)

---

### Approach 3: Multi-Point Prompting (FG + BG)
Two foreground points + one background point with mask refinement. Improves over single point but less precise than bounding boxes.

![Multi-Point](plot%20results/segm3.png)

---

### Example Cases

**Good case:** predicted mask closely matches ground truth.

![Good Example](plot%20results/qual2.png)

**Suboptimal case:** predicted mask misses parts of lungs and includes some false positives.

![Suboptimal Example](plot%20results/qual1.png)

---

## 📈 Results & Evaluation

The three prompting strategies were evaluated on the validation set using the F1-score metric.

| Approach | Mean F1 Score | Standard Deviation |
|--------|---------------|--------------------|
| Single Point (FG + BG) | 0.5443 | 0.1461 |
| Bounding Boxes | **0.8567** | 0.0339 |
| Multi-Point (FG + BG) | 0.8169 | 0.0690 |

---

## 🧪 Qualitative Analysis

Analysis of the best-performing method (bounding boxes):

**Good cases:**
- Predicted masks closely match the ground truth with smooth and consistent lung boundaries.

**Suboptimal cases:**
- In low-contrast images, the model may miss parts of the lower lung lobes or include false positives in the abdominal region.

Visualization examples are available in the `plots/task2/` directory.

---

## 🛠️ Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/yourusername/sam-lung-segmentation.git
cd sam-lung-segmentation
```

### Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib jupyter
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Run the Notebook

Open `sam_lung_xrays_1.ipynb` to view the full implementation and visualization pipeline.

---

## 🎓 Credits

- **Course:** Advanced Deep Learning (2025), University of Copenhagen (UCPH)
- **Original Notebook Authors:** Jakob Ambsdorf, Mathias Perslev, Stefan Sommer
- **Implementation:** Your Name
- **Model:** Segment Anything Model (SAM), Meta AI
- **Data Source:** van Ginneken et al., *Medical Image Analysis* (2006)

This project is for educational purposes only.
