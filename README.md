# CVSS 2.0 → CVSS 3.x Neural Network Converter

This repository contains a machine-learning–based converter that automatically transforms CVSS v2.0 vulnerability metrics into CVSS v3.x base metrics and computes the final CVSS Base Score (3.1).  
The project uses eight independent neural network models (one per each CVSS v3.x metric) trained on over 73k real vulnerability records.

The main goal of this project is to provide a fully automated and consistent way to convert historical CVSS 2.0 data into the newer CVSS 3.x standard — something that does not exist officially.

---

## Project Overview

CVSS v2.0 and CVSS v3.x differ significantly in structure, metrics and interpretation.  
Because there is no official conversion method, organizations face issues when comparing or aggregating vulnerability data across systems and time.

This project solves the problem by training neural network classifiers to predict:

- AV – Attack Vector  
- AC – Attack Complexity  
- PR – Privileges Required  
- UI – User Interaction  
- S – Scope  
- C – Confidentiality Impact  
- I – Integrity Impact  
- A – Availability Impact  

Once predicted, all metrics are combined and processed using the official CVSS 3.1 formula to compute the final Base Score.

---

## Machine Learning Architecture

Each of the eight models is a separate MLP (Multilayer Perceptron):

- **Input:** 57 numerical features (50 text-based CVE features + CVSS 2.0 fields)
- **Hidden layers:**  
  - Dense(64, ReLU)  
  - Dense(32, ReLU)
- **Output:** Softmax layer adjusted to the number of classes for each metric
- **Loss function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam`
- **Training:** 20 epochs with `EarlyStopping` and 20% validation split  
- **Class balancing:** SMOTE oversampling for minority classes

All models achieve **over 90% accuracy**, with Scope (S) and User Interaction (UI) exceeding 96%.

---

## Dataset

The dataset contains **73,179 vulnerability records**, with:

- 50 text-based features (keyword frequencies)
- CVSS 2.0 metrics
- CVSS 3.x metrics
- Final Base Score (v3.x)

The dataset itself is not included in this repository for licensing reasons.

---

## Base Score Calculation

The converter:

1. Loads the trained neural network models  
2. Predicts all eight CVSS v3.x metrics  
3. Converts class indices → metric letters → numerical weights  
4. Computes the final Base Score using the official CVSS 3.1 formula  
5. Produces the final CVSS vector string

Example output:

