# Internship Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)

## 📌 Project Overview
This project delivers a comprehensive, production-grade continuous regression pipeline developed on the **California Housing Dataset** (20,640 census block records). The project systematically implements:
1. **Missing Value Imputation**: Median imputation for `total_bedrooms` (~1.0% missingness) to ensure numerical robustness without data leakage.
2. **Domain-Specific Feature Engineering & Interaction Modeling**:
   - Structural ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
   - Affluence-space interaction feature (`income_rooms_interaction = median_income * rooms_per_household`).
   - Spatial proximity features computing Euclidean distances to California's primary metropolitan hubs (Los Angeles and San Francisco) and minimum metro distance (`min_dist_to_metro`).
3. **Multicollinearity Diagnosis**: Formal calculation of Variance Inflation Factors (VIF) diagnosing severe collinearity in raw room and household counts ($\text{VIF} > 28$).
4. **Data Preprocessing**: Strict leakage-free `ColumnTransformer` pipeline employing `StandardScaler` on continuous features and `OneHotEncoder(drop='first')` on categorical attributes (`ocean_proximity`).
5. **Hyperparameter Optimization via 5-Fold `GridSearchCV`**: Log-scale parameter sweeps across $\alpha \in [0.01, 1000.0]$ for $L_2$ Ridge and $L_1$ Lasso regression optimizing Root Mean Squared Error.
6. **Rigorous Model Benchmarking**: Comparative performance evaluation across **RMSE, MAE, and $R^2$** on both training and holdout test partitions (80/20 split), backed by 5-fold cross-validation.
7. **Comprehensive Residual Error Analysis**: 45-degree parity plots ($y = \hat{y}$), residual distribution KDEs, homoscedasticity checks, and analysis of the $500,001 census price cap.

---

## 📂 Project Structure & Deliverables

| File / Folder | Type | Description |
| :--- | :---: | :--- |
| **[`Task_2_California_Housing_Regression.ipynb`](Task_2_California_Housing_Regression.ipynb)** | **Primary Deliverable** | Standalone Jupyter Notebook with 29 cells, comprehensive markdown theory, mathematical formulations, executed Python code, metrics tables, and pre-rendered inline plots. |
| **[`housing.csv`](housing.csv)** | Dataset | Benchmark California housing dataset (20,640 records, 10 attributes). |
| **[`build_and_run_task2.py`](build_and_run_task2.py)** | Pipeline Script | Executable Python script reproducing the complete training pipeline, model tuning, and visualization exports. |
| **[`generate_notebook.py`](generate_notebook.py)** | Generator Script | Programmatic generator that constructs and pre-renders the Jupyter Notebook with embedded Base64 figures. |
| **[`visualizations/`](visualizations/)** | Directory | High-resolution exported diagnostic plots: |
| ├── `california_housing_spatial.png` | Plot | Geographical bubble scatter plot colored by house value and scaled by population. |
| ├── `correlation_heatmap.png` | Plot | Triangular correlation matrix heatmap across original and engineered features. |
| ├── `alpha_tuning_curves.png` | Plot | 5-Fold CV RMSE vs. $\alpha$ search curves for Ridge and Lasso on semi-log scale. |
| ├── `actual_vs_predicted.png` | Plot | 3-panel Actual vs. Predicted 45-degree parity plots for OLS, Ridge, and Lasso. |
| ├── `residual_analysis.png` | Plot | 6-panel residual diagnostics (Residuals vs. Fitted and Error Distribution KDEs). |
| └── `coefficient_comparison.png` | Plot | Standardized coefficient shrinkage comparison across OLS, Ridge, and Lasso. |

---

## 🔬 Mathematical Formulations

### 1. Ordinary Least Squares (OLS)
OLS minimizes the unpenalized sum of squared residuals:
$$\min_w \mathcal{L}_{\text{OLS}}(w) = \frac{1}{2n} \|y - Xw\|_2^2 = \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T w)^2$$
Closed-form solution:
$$\hat{w}_{\text{ols}} = (X^T X)^{-1} X^T y$$

### 2. Ridge Regression ($L_2$ Regularization)
Ridge introduces a quadratic penalty that controls the $L_2$ Euclidean norm of the weight vector:
$$\min_w \mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_2^2 = \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T w)^2 + \alpha \sum_{j=1}^p w_j^2$$
Closed-form solution:
$$\hat{w}_{\text{ridge}} = (X^T X + 2n\alpha I)^{-1} X^T y$$
*Key Property*: Adding $2n\alpha I$ to the Gram matrix $X^T X$ increases all eigenvalues by a positive constant, guaranteeing invertibility and shrinking collinear coefficients uniformly.

### 3. Lasso Regression ($L_1$ Regularization)
Lasso imposes an $L_1$ norm penalty that promotes parameter sparsity:
$$\min_w \mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_1 = \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T w)^2 + \alpha \sum_{j=1}^p |w_j|$$
Optimized via coordinate descent using the soft-thresholding operator:
$$S_{\alpha}(z) = \text{sign}(z) \max(|z| - \alpha, 0)$$
*Key Property*: The sharp corners of the $L_1$ constraint region at coordinate axes drive uninformative or redundant collinear coefficients **exactly to zero**, performing automated feature selection.

### 4. Multicollinearity & Variance Inflation Factor (VIF)
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
Where $R_j^2$ is the coefficient of determination from regressing feature $x_j$ against all other remaining features $X_{-j}$. $\text{VIF} > 10$ indicates severe multicollinearity that destabilizes OLS.

---

## 📊 Benchmark Results

### 1. Model Performance Summary
Evaluated on an 80/20 holdout test split (16,512 train samples, 4,128 test samples) and verified via 5-Fold Cross-Validation:

| Metric | OLS Regression | Ridge Regression ($\alpha=0.5$) | Lasso Regression ($\alpha=5.0$) |
| :--- | :---: | :---: | :---: |
| **Optimal Regularization $\alpha$** | — | **0.5** | **5.0** |
| **5-Fold CV RMSE ($)** | **\$66,883.95** ($\pm \$1,536$) | **\$66,882.80** ($\pm \$1,538$) | **\$66,883.50** ($\pm \$1,536$) |
| **Training RMSE ($)** | \$66,572.28 | \$66,572.75 | \$66,573.08 |
| **Holdout Test RMSE ($)** | **\$70,800.49** | **\$70,806.70** | **\$70,801.12** |
| **Training MAE ($)** | \$48,155.08 | \$48,156.13 | \$48,154.82 |
| **Holdout Test MAE ($)** | **\$49,805.53** | **\$49,811.42** | **\$49,807.14** |
| **Training $R^2$** | **0.6685** | **0.6685** | **0.6685** |
| **Holdout Test $R^2$** | **0.6175** | **0.6174** | **0.6175** |

### 2. Multicollinearity Audit (VIF Table)
Before feature engineering, raw aggregate features suffered from extreme multicollinearity:

| Feature | Raw VIF | Status |
| :--- | :---: | :--- |
| `households` | **28.28** | 🚨 Severe Multicollinearity ($\text{VIF} > 10$) |
| `total_bedrooms` | **26.88** | 🚨 Severe Multicollinearity ($\text{VIF} > 10$) |
| `total_rooms` | **12.13** | 🚨 Severe Multicollinearity ($\text{VIF} > 10$) |
| `latitude` | **8.83** | ⚠️ Moderate Multicollinearity |
| `longitude` | **8.70** | ⚠️ Moderate Multicollinearity |
| `population` | **6.26** | ⚠️ Moderate Multicollinearity |
| `median_income` | **1.69** | ✅ Low / Independent |
| `housing_median_age` | **1.26** | ✅ Low / Independent |

*Engineering Solution*: Replacing raw counts with normalized ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) and applying regularized estimators (Ridge/Lasso) prevents matrix ill-conditioning.

### 3. Top Standardized Predictors (Model Weights)

| Rank | Feature | Standardized Weight | Economic / Domain Meaning |
| :---: | :--- | :---: | :--- |
| 1 | `median_income` | **+\$84,809** | Block group purchasing power is the primary driver of property valuation. |
| 2 | `ocean_proximity_ISLAND` | **+\$145,000** | Extreme scarcity premium for island locations. |
| 3 | `ocean_proximity_NEAR OCEAN`| **+\$22,880** | Proximity to coastline commands a significant price premium. |
| 4 | `bedrooms_per_room` | **+\$15,335** | High living-space-to-bedroom ratio indicates larger, luxury residences. |
| 5 | `min_dist_to_metro` | **-\$13,757** | Distance from LA / SF economic centers creates substantial negative valuation decay. |
| 6 | `ocean_proximity_INLAND` | **-\$32,308** | Inland properties trade at a massive discount compared to coastal areas. |
| 7 | `population` | **-\$46,999** | High raw block population indicates overcrowding and lower per-capita valuation. |

---

## 🔍 Residual Diagnostics & Insights

1. **Parity Check ($y = \hat{y}$)**:
   - All three models follow the 45-degree parity line closely across the $\$100,000$ to $\$400,000$ valuation range.
2. **Census Cap Ceiling Effect**:
   - The California census dataset caps median house values at **$\$500,001$**. In the residual vs. fitted plots, this manifests as a crisp diagonal upper boundary ($500,000 - \hat{y}$), indicating right-censored data rather than model misspecification.
3. **Error Distribution**:
   - Residual distribution curves are centered symmetrically around $0$ with approximately normal tails, satisfying core linear regression diagnostic assumptions.

---

## ⚙️ Environment Setup & Reproduction

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

### 2. Running the Interactive Jupyter Notebook
Open the deliverable notebook directly in VS Code, Jupyter Notebook, or JupyterLab:
```bash
jupyter notebook Task_2_California_Housing_Regression.ipynb
```
*Note: The notebook is completely pre-executed. All metrics, outputs, and plots will render immediately.*

### 3. Running via Python Script
To execute the pipeline and re-generate all visualization plots programmatically:
```bash
python build_and_run_task2.py
```
