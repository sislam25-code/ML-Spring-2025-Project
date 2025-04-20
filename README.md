#  NY‑DonorsChoose Funding Risk Prediction

> A machine-learning pipeline using decision trees and XGBoost to identify the 10% of **New York State–filtered** DonorsChoose projects least likely to receive full funding—guiding equitable resource allocation in K‑12 education.

---

## Overview --> need to work on this later

Funding disparities across New York public schools leave under-resourced classrooms at a disadvantage. This project analyzes a **New York State subset** of DonorsChoose data to build a binary classification task (`fully_funded` = 0/1) that flags underfunded projects. Specifically, the pipeline:

- Defines a binary classification task on the filtered dataset
- Trains interpretable tree-based models to predict the top 10% most at-risk proposals
- Provides ranked insights for targeted policy and donor action

---

## Installation & Setup

### Download Code & Data

Download the project ZIP from GitHub:

- Click the green **Code** button on the repository’s main page → **Download ZIP**
- Unzip into your working directory

The `Data/` folder includes:

- **Raw data:** original DonorsChoose CSV files (`projects.csv`, `outcomes.csv`)
- **Cleaned data:** outputs from Stage 1 EDA
- **Model inputs:** feature sets used for model training

### Environment & Libraries

This project is best run in a Jupyter Notebook or Spyder within an Anaconda environment for smooth package management.

#### Install Required Packages —> add more packages needed

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn xgboost
from sklearn.model_selection import train_test_split
```

---

## &#x20;Project Stages & Notebooks

| Stage   | Notebook                     | Purpose                                                    |
| ------- | ---------------------------- | ---------------------------------------------------------- |
| Stage 1 | `Stage1_EDA.ipynb`           | Filter NY projects, clean data, EDA, and visualization     |
| Stage 2 | `Stage2_Preprocessing.ipynb` | Feature creation, encoding, and train/test splits          |
| Stage 3 | `Stage3_Modeling.ipynb`      | Train Decision Tree & XGBoost (5-fold CV), evaluate & rank |

All data files reside in the `Data/` folder.

---

## Results & Findings

| Metric (XGBoost)        | Score |
| ----------------------- | ----- |
| Recall (unfunded class) |       |
| Precision               |       |
| AUPRC                   |       |

-  add findins here 
-

---

## &#x20;Reproduction Steps

Run each notebook in order to reproduce the analysis and results:

1. **Stage 1 – Data Cleaning & EDA:** Run `Stage1_EDA.ipynb` to filter, clean, and explore the raw data.
2. **Stage 2 – Preprocessing & Feature Engineering:** Run `Stage2_Preprocessing.ipynb` to generate modeling features and train/test splits.
3. **Stage 3 – Modeling & Evaluation:** Run `Stage3_Modeling.ipynb` to train the decision tree and XGBoost models, evaluate performance, and generate risk scores.

Ensure the `Data/` folder is present before running the notebooks.

---

## Contributors

- **Stage 1 (Data Cleaning & EDA): Liufei Chen**
- **Stage 2 (Preprocessing & Features): Coby Dodson**
- **Stage 3 (Modeling & Evaluation): Samiha Islam**

---




