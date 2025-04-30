#  NY‑DonorsChoose Funding Risk Prediction

> A machine-learning pipeline using decision trees and random forest modeling to identify the 10% of **New York State–filtered** DonorsChoose projects least likely to receive full funding—guiding equitable resource allocation in K‑12 education.

---

## Overview

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

#### Install Required Packages

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from scipy.stats import pointbiserialr
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

---

## &#x20;Project Stages & Notebooks

| Stage   | Notebook                     | Purpose                                                    |
| ------- | ---------------------------- | ---------------------------------------------------------- |
| Stage 1 | `Stage1_EDA.ipynb`           | Filter NY projects, clean data, EDA, and visualization     |
| Stage 2 | `Stage2_Preprocessing.ipynb` | Feature creation, encoding, and train/test splits          |
| Stage 3 | `Stage3_Modeling.ipynb`      | Train Random Forest Decision Tree & XGBoost (5-fold CV), evaluate & rank |

All data files reside in the `Data/` folder.

---

##  Model Results

Baseline
| Sampling Strategy | Model          | Precision (0) | Recall (0) |
|------------------|---------------|--------------|-----------|
| SMOTE (Oversampling) | Random Forest | 0.52         | 0.31      |
| SMOTE (Oversampling) | Decision Tree  | 0.41         | 0.40      |
| RUS (Undersampling)  | Random Forest | 0.40         | 0.71      |
| RUS (Undersampling)  | Decision Tree  | 0.36         | 0.66      |

3-Kfold Cross Validation
| Sampling Strategy | Model          | Precision (0) | Recall (0) |
|------------------|---------------|--------------|-----------|
| SMOTE (Oversampling) | Random Forest | 0.89             |  0.84      |
| SMOTE (Oversampling) | Decision Tree  | 0.83          | 0.82      |
| RUS (Undersampling)  | Random Forest |  0.71             | 0.73      |
| RUS (Undersampling)  | Decision Tree  | 0.67          | 0.68      |


- We found that a 3-Kfold Cross Validation model of a random forest classfier using SMOTE oversampling produced the highest precision and recall scores out of all models
- Unfortunately, running this model took 40+ minutes, so we built a simpler version, which we refer to as a Minimal Random Forest Model, to inform our results for this project
- The Minimal Random Forest Model trained on only 1% of the data, resampled with SMOTE on only 50% of the minority class to generate fewer additional datapoints, and restricted the number of trees to 10. As such, our precision and recall scores were lower than the previously defined models. In the real world, we would advise anyone running this model to use a machine powerful enough to use the original Kfold random forest model with SMOTE oversampling, rather than the Minimal Random Forest

##  Model Interpretation
We found that the top 5 features predicting project underfunding are:
- students_reached_capped with a correlation of 0.28
- school_zip with a correlation of 0.053
- price_capped_in with a correlation of 0.8
- eligible_almost_home_match with a correlation of -.071
- price_capped_ex with a correlation of 0.8
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




