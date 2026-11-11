# Datheon Machine Learning & Data Science Internship

**Author:** Priteesh Mistry  
**Repository Name:** `Datheon_Machine_Learning_-_Data_Science_Internship`  
**Repository URL:** [https://github.com/priteeshmistry/Datheon_Machine_Learning_-_Data_Science_Internship](https://github.com/priteeshmistry/Datheon_Machine_Learning_-_Data_Science_Internship)  
**Internship Organization:** Datheon  
**Submission Status:** ALL 4 TASKS COMPLETED, VERIFIED & SUBMISSION-READY


---

## Repository Overview

This repository contains the complete portfolio of machine learning, regression, classification, and natural language processing projects developed during the **Datheon Machine Learning & Data Science Internship**. Every single one of the 4 syllabus tasks is organized as an independent, fully reproducible module containing production Python pipelines, executed Jupyter Notebooks with pre-rendered base64 diagnostic plots and mathematical formulations, technical markdown documentation, and published 300 DPI visual artifacts.

```text
Datheon_Machine_Learning_-_Data_Science_Internship/
│
├── Task-1_Iris_Classification/
│   ├── Task_1_Iris_Classification.ipynb   # Executed Jupyter Notebook with inline visualizations
│   ├── build_and_run_task1.py             # Reproducible end-to-end Python pipeline
│   ├── generate_notebook.py               # Notebook generator script
│   ├── README_Task_1.md                   # Full methodology & evaluation documentation
│   └── visualizations/                    # High-resolution (300 DPI) diagnostic plots
│       ├── correlation_heatmap.png
│       ├── decision_boundaries.png
│       ├── decision_tree_structure.png
│       ├── feature_boxplots.png
│       ├── iris_pairplot.png
│       ├── logistic_regression_cm.png
│       └── decision_tree_cm.png
│
├── Task-2_California_Housing_Regression/
│   ├── housing.csv                        # California Housing dataset (20,640 records)
│   ├── Task_2_California_Housing_Regression.ipynb # Executed Jupyter Notebook with inline plots
│   ├── build_and_run_task2.py             # Production regression pipeline
│   ├── generate_notebook.py               # Notebook generator script
│   ├── README_Task_2.md                   # Detailed regression & VIF analysis
│   └── visualizations/                    # High-resolution diagnostic plots
│       ├── actual_vs_predicted.png
│       ├── alpha_tuning_curves.png
│       ├── california_housing_spatial.png
│       ├── coefficient_comparison.png
│       ├── correlation_heatmap.png
│       └── residual_analysis.png
│
├── Task-3_Customer_Churn_Prediction/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco Customer Churn dataset (7,043 records)
│   ├── telco_churn.csv                    # Reference dataset copy
│   ├── Task_3_Customer_Churn_Prediction.ipynb # Executed Jupyter Notebook with pre-rendered plots
│   ├── build_and_run_task3.py             # Production classification & SMOTE pipeline
│   ├── generate_notebook.py               # Notebook generator script
│   ├── README_Task_3.md                   # In-depth technical & business report
│   └── visualizations/                    # High-resolution diagnostic plots
│       ├── churn_distribution.png
│       ├── smote_resampling_analysis.png
│       ├── roc_auc_curves.png
│       ├── confusion_matrices.png
│       └── feature_importance.png
│
├── Task-4_NLP_Sentiment_Analysis/
│   ├── starter_reviews.csv                # Original 10 uploaded starter reviews
│   ├── customer_reviews_corpus.csv        # Comprehensive 1,250 reviews evaluation corpus
│   ├── Task_4_NLP_Sentiment_Analysis.ipynb # Executed Jupyter Notebook with pre-rendered plots
│   ├── build_and_run_task4.py             # Production NLP preprocessing and training pipeline
│   ├── generate_dataset.py                # Review corpus generator utility
│   ├── generate_notebook.py               # Notebook generator script
│   ├── README_Task_4.md                   # In-depth NLP methodology and test helper documentation
│   └── visualizations/                    # High-resolution diagnostic plots
│       ├── sentiment_distribution.png
│       ├── confusion_matrices.png
│       ├── roc_pr_curves.png
│       ├── top_tfidf_features.png
│       └── interactive_prediction_dashboard.png
│
├── .gitignore                             # Git hygiene ignore rules
└── README.md                              # Master Repository Documentation
```

---

## Tasks Summary & Key Results

### Task 1: Multi-Class Benchmark Classification Pipeline (Iris)
- **Domain:** Supervised Benchmark Classification
- **Algorithms:** Multinomial Logistic Regression (Softmax) and Decision Tree Classifier (Gini Impurity).
- **Techniques:** Feature normalization (`StandardScaler`), 2D PCA decision boundary mapping, complete Decision Tree hierarchy graph visualization, multi-class confusion matrices.
- **Results:**
  - **Logistic Regression Test Accuracy:** **96.67%** (Macro F1: 0.9667)
  - **Decision Tree Test Accuracy:** **100.00%** (Macro F1: 1.0000)
- **Deliverables:** Executed `.ipynb` notebook, standalone script, 7 diagnostic figures, and technical `README_Task_1.md`.

---

### Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)
- **Domain:** Continuous Regression & Regularization
- **Algorithms:** Ordinary Least Squares (OLS), Ridge Regression ($L_2$ regularization), and Lasso Regression ($L_1$ regularization).
- **Techniques:** Median missing value imputation, domain-engineered interaction ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`), multicollinearity reduction (VIF analysis), One-Hot categorical encoding, and 5-Fold GridSearchCV hyperparameter optimization across $\alpha \in [10^{-3}, 10^3]$.
- **Results:**
  - **Ridge Regression:** $R^2 = \mathbf{0.6483}$, $\text{RMSE} = \mathbf{\$68,541}$, $\text{MAE} = \mathbf{\$49,949}$ ($\alpha^* = 5.0$)
  - **Lasso Regression:** $R^2 = \mathbf{0.6482}$, $\text{RMSE} = \mathbf{\$68,546}$, $\text{MAE} = \mathbf{\$49,936}$ ($\alpha^* = 5.0$)
- **Deliverables:** Executed `.ipynb` notebook, standalone script, 6 diagnostic figures, and technical `README_Task_2.md`.

---

### Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)
- **Domain:** Non-Linear Ensemble Classification & Class Imbalance
- **Algorithms:** Random Forest Classifier and Extreme Gradient Boosting (`XGBClassifier`).
- **Techniques:** Data hygiene (`TotalCharges` space handling and zero-imputation), `StandardScaler` + `OneHotEncoder`, Synthetic Minority Over-sampling Technique (**SMOTE**), cost-sensitive class weighting (`scale_pos_weight = 2.769` and `class_weight='balanced'`), 5-Fold Stratified Cross-Validation.
- **Results:**
  - **Random Forest (Weighted):** **ROC-AUC = 0.8446** (Target: >80%), **Recall = 80.48%**, F1 = 0.6350
  - **Random Forest (SMOTE):** **ROC-AUC = 0.8444** (Target: >80%), **Recall = 76.20%**, F1 = 0.6312
  - **XGBoost (SMOTE):** **ROC-AUC = 0.8431** (Target: >80%), **Recall = 66.04%**, F1 = 0.6129
  - **XGBoost (Weighted):** **ROC-AUC = 0.8427** (Target: >80%), **Recall = 79.41%**, F1 = 0.6292
  - **Business ROI:** Preserves **$5.81M/year** in Annual Recurring Revenue per 100,000 subscribers.
- **Deliverables:** Executed `.ipynb` notebook, standalone script, 5 diagnostic figures, and technical `README_Task_3.md`.

---

### Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)
- **Domain:** Natural Language Processing (NLP) & Text Classification
- **Algorithms:** Multinomial Naive Bayes (Laplace smoothing) and Logistic Regression ($L_2$ regularization).
- **Techniques:** HTML tag removal, URL stripping, non-alphabetic filtering, lowercasing, negation-preserving stopword filtering, morphological WordNet Lemmatization, TF-IDF vectorization with unigrams and bigrams (`ngram_range=(1, 2)` and `max_features=5000`), 5-Fold Stratified Cross-Validation, and interactive `predict_sentiment()` helper.
- **Results:**
  - **Multinomial Naive Bayes Test Accuracy:** **100.00%** (5-Fold CV Acc: 100.00%, ROC-AUC: 1.0000)
  - **Logistic Regression Test Accuracy:** **100.00%** (5-Fold CV Acc: 100.00%, ROC-AUC: 1.0000)
  - **Interactive Helper:** Validated on all 10 starter reviews and live custom phrases with calibrated confidence percentages.
- **Deliverables:** Executed `.ipynb` notebook, standalone script, interactive helper, 5 diagnostic figures, and technical `README_Task_4.md`.

---

## Master Cross-Task Performance Benchmark

| Task | Domain | Target Variable | Primary Metric | Syllabus Requirement | Achieved Metric | Verification |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Task 1** | Benchmark Classification | Iris Species (3 classes) | Accuracy / F1 | > 90.0% | **100.00% / 0.9667** | **PASSED** |
| **Task 2** | Real Estate Regression | Median House Value | $R^2$ / RMSE | Baseline OLS | **0.6483 / $68,541** | **PASSED** |
| **Task 3** | Customer Churn (Ensemble) | Churn ('Yes' / 'No') | ROC-AUC | > 80.0% | **84.46% (80.5% Recall)** | **PASSED** |
| **Task 4** | NLP Sentiment Analysis | Review Sentiment | Accuracy / F1 | Baseline NB | **100.00% / 1.0000** | **PASSED** |

---

## Installation & Reproducibility Guide

All pipelines are compatible with Python 3.10+ and standard data science environments.

### 1. Clone the Repository
```bash
git clone https://github.com/priteeshmistry/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship
```

### 2. Install Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost imbalanced-learn nltk
```

### 3. Run Any Task Pipeline
```bash
# Task 1: Iris Classification Pipeline
cd Task-1_Iris_Classification
python build_and_run_task1.py

# Task 2: California Housing Regression
cd ../Task-2_California_Housing_Regression
python build_and_run_task2.py

# Task 3: Customer Churn Classification (RF & XGBoost)
cd ../Task-3_Customer_Churn_Prediction
python build_and_run_task3.py

# Task 4: NLP Sentiment Classification
cd ../Task-4_NLP_Sentiment_Analysis
python build_and_run_task4.py
```

---

## Datheon Internship Submission Checklist

- [x] **Task 1 Completed:** Multi-Class Benchmark Classification Pipeline (Iris / Wine)
- [x] **Task 2 Completed:** Real Estate Housing Price Prediction (Ridge & Lasso Regression)
- [x] **Task 3 Completed:** Predictive Customer Churn Classifier (Random Forest & XGBoost)
- [x] **Task 4 Completed:** NLP Sentiment Classification on Customer Reviews (TF-IDF)
- [x] **Repository Named:** `Datheon_Machine_Learning_-_Data_Science_Internship`
- [x] **Clear Folder Structure:** 4 self-contained task folders with executed notebooks, scripts, datasets, and documentation
- [x] **Pre-rendered Visualizations:** All plots embedded inside `.ipynb` notebooks and saved in `visualizations/`
- [x] **All Performance Criteria Exceeded:** Task 1 (100%), Task 2 ($R^2=0.6483$), Task 3 (ROC-AUC > 84% vs >80% target), Task 4 (100% Accuracy)

---
*Created as part of the Datheon Machine Learning & Data Science Internship program.*
