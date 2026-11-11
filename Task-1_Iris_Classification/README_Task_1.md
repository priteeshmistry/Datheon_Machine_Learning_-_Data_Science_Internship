# Internship Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)

## 📌 Project Overview
This project delivers an end-to-end, production-grade supervised machine learning classification pipeline benchmarked on Fisher's Iris Dataset. The pipeline implements:
1. **Exploratory Data Analysis (EDA)** with Seaborn pairplots, correlation matrix heatmaps, and distribution boxplots.
2. **Data Preprocessing & Feature Scaling** using `StandardScaler` ($z$-score normalization) and stratified 80/20 train/test splitting.
3. **Supervised Model Training** using two distinct classification paradigms:
   - **Logistic Regression**: Linear parametric classifier optimized via multinomial cross-entropy with $L_2$ regularization.
   - **Decision Tree Classifier**: Non-parametric tree model utilizing Gini Impurity splits with depth regularization (`max_depth=3`).
4. **Comprehensive Benchmarking** across Accuracy, Precision, Recall, and F1-Score (both Macro and Weighted averages), backed by 5-fold Stratified Cross-Validation.
5. **Diagnostic Visualizations**:
   - Side-by-side Confusion Matrix heatmaps.
   - 2D Decision Boundary plots contrasting smooth linear hyperplanes vs orthogonal axis-aligned partitions.
   - Visual Decision Tree diagram and Gini Feature Importance ranking.

---

## 📂 Project Structure & Deliverables

| File / Folder | Description |
| :--- | :--- |
| **`Task_1_Iris_Classification.ipynb`** | **Primary Deliverable**: Standalone Jupyter Notebook containing structured markdown cells, fully executed Python code, metrics outputs, and high-resolution plots. |
| **`iris.csv`** | Raw benchmark dataset (150 rows, 4 continuous features, 3 balanced flower species). |
| **`build_and_run_task1.py`** | Standalone executable Python pipeline script for automated training, evaluation, and plot exports. |
| **`visualizations/`** | Directory containing all exported high-resolution PNG plots: |
| ├── `confusion_matrices.png` | Side-by-side Confusion Matrix heatmaps for both models. |
| ├── `decision_boundaries.png` | 2D Decision Boundary comparison plot (Petal Length vs Petal Width). |
| ├── `pairplot.png` | Seaborn pairplot with KDE density estimates across all features. |
| ├── `correlation_matrix.png` | Triangular correlation heatmap. |
| ├── `feature_distributions.png`| Boxplots showing feature distribution per class. |
| ├── `decision_tree_structure.png`| Pruned Decision Tree diagram displaying node splits, Gini values, and sample counts. |
| └── `feature_importance.png` | Gini Feature Importance horizontal bar chart. |

---

## ⚙️ Environment Setup & Reproduction

### 1. Requirements
Ensure Python 3.10+ is installed along with the required libraries:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

### 2. Running the Jupyter Notebook
Open the notebook in Visual Studio Code, Jupyter Notebook, or JupyterLab:
```bash
jupyter notebook Task_1_Iris_Classification.ipynb
```
*Note: The notebook is pre-executed and all cell outputs and charts will display immediately without needing a manual re-run.*

### 3. Running via Python Script
To execute the pipeline and re-generate all visualization plots programmatically:
```bash
python build_and_run_task1.py
```

---

## 📊 Benchmark Results

### 1. Model Performance Summary
Evaluated on the 20% holdout test set (30 samples: 10 Setosa, 10 Versicolor, 10 Virginica):

| Metric | Logistic Regression | Decision Tree (`max_depth=3`) |
| :--- | :---: | :---: |
| **5-Fold Cross-Validation Accuracy** | **95.83%** ($\pm 2.64\%$) | **95.83%** ($\pm 2.64\%$) |
| **Holdout Test Accuracy** | **93.33%** | **96.67%** |
| **Macro Precision** | **93.33%** | **96.97%** |
| **Macro Recall** | **93.33%** | **96.67%** |
| **Macro F1-Score** | **93.33%** | **96.66%** |
| **Weighted F1-Score** | **93.33%** | **96.66%** |

### 2. Classification Reports Breakdown

#### Logistic Regression:
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

#### Decision Tree (`max_depth=3`):
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

---

## 🔬 In-Depth Engineering & Technical Insights

1. **Why `StandardScaler` Matters**:
   - In **Logistic Regression**, gradient-based optimizers (L-BFGS) navigate ellipsoidal loss contours much faster and without oscillating when features are centered at mean 0 and scaled to unit variance. Additionally, the $L_2$ regularization penalty ($\lambda \sum w_j^2$) assumes all weights operate on equivalent scales.
   - For **Decision Trees**, normalization is redundant because tree split decisions depend strictly on ordinal ranking ($x_j \le \theta$). Monotonic scaling transformations preserve rank order identically.

2. **Decision Boundary Mechanics**:
   - The **Decision Tree** achieves higher test accuracy ($96.67\%$) because its axis-aligned orthogonal partitions readily carve out rectangular boxes around clusters, whereas **Logistic Regression**'s linear hyperplanes struggle with the non-linear overlap between *Versicolor* and *Virginica*.

3. **Feature Importance**:
   - Decision Tree analysis proves that `petal_width` ($56.2\%$) and `petal_length` ($42.2\%$) account for over $98\%$ of discriminatory capacity, rendering sepal features largely non-essential for classifying these three species.
