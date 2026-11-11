import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

WORKSPACE = r"D:\Datheon Internship\Task-2_California_Housing_Regression"
CSV_PATH = os.path.join(WORKSPACE, "housing.csv")
OUTPUT_DIR = os.path.join(WORKSPACE, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("TASK 2: REAL ESTATE HOUSING PRICE PREDICTION (OLS, RIDGE & LASSO REGRESSION)")
print("=" * 80)

# ---------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------
print("\n[Step 1] Loading California Housing Dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nDataset Info Summary:")
print(df.info())
print("\nFirst 5 Records:")
print(df.head())

print("\nNull Values Check:")
null_summary = df.isnull().sum()
print(null_summary[null_summary > 0])

# ---------------------------------------------------------
# Step 2: Exploratory Spatial Plot & Missing Imputation
# ---------------------------------------------------------
print("\n[Step 2] Generating Exploratory Spatial Distribution Plot...")
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['longitude'], df['latitude'],
    c=df['median_house_value'], cmap='coolwarm',
    s=df['population'] / 100, alpha=0.5, edgecolors='none'
)
cbar = plt.colorbar(scatter)
cbar.set_label('Median House Value ($)', fontsize=11, weight='bold')
plt.title('California Housing: Geographical Distribution by Median House Value', fontsize=13, weight='bold', pad=12)
plt.xlabel('Longitude', fontsize=11)
plt.ylabel('Latitude', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "california_housing_spatial.png"), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Step 3: Missing Value Imputation & Feature Engineering
# ---------------------------------------------------------
print("\n[Step 3] Imputing Missing Values & Engineering Interaction Features...")

# Median imputation for total_bedrooms (preserving original median)
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
print(f"Imputed missing 'total_bedrooms' using median value: {median_bedrooms}")

# 1. Structural Ratio Features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# 2. Interaction Features
# Wealth/living space interaction
df['income_rooms_interaction'] = df['median_income'] * df['rooms_per_household']

# 3. Metropolitan Economic Proximity Features
# Los Angeles coordinates: (34.0522, -118.2437); San Francisco coordinates: (37.7749, -122.4194)
df['dist_to_la'] = np.sqrt((df['latitude'] - 34.0522)**2 + (df['longitude'] - -118.2437)**2)
df['dist_to_sf'] = np.sqrt((df['latitude'] - 37.7749)**2 + (df['longitude'] - -122.4194)**2)
df['min_dist_to_metro'] = df[['dist_to_la', 'dist_to_sf']].min(axis=1)

print("Engineered 6 new domain & interaction features:")
print("- rooms_per_household = total_rooms / households")
print("- bedrooms_per_room = total_bedrooms / total_rooms")
print("- population_per_household = population / households")
print("- income_rooms_interaction = median_income * rooms_per_household")
print("- dist_to_la = Euclidean distance to Los Angeles center")
print("- dist_to_sf = Euclidean distance to San Francisco center")
print("- min_dist_to_metro = min(dist_to_la, dist_to_sf)")

# Correlation Heatmap
print("\n[Step 4] Generating Correlation Heatmap...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r',
    vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8}
)
plt.title('California Housing Correlation Matrix (Original & Engineered Features)', fontsize=14, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Step 5: Multicollinearity Diagnosis (VIF Analysis)
# ---------------------------------------------------------
print("\n[Step 5] Analyzing Multicollinearity via Variance Inflation Factor (VIF)...")

def compute_vif(X_df):
    vifs = []
    for col in X_df.columns:
        y_c = X_df[col]
        X_c = X_df.drop(columns=[col])
        lr = LinearRegression().fit(X_c, y_c)
        r2 = lr.score(X_c, y_c)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vifs.append({'Feature': col, 'VIF': vif})
    return pd.DataFrame(vifs).sort_values('VIF', ascending=False).reset_index(drop=True)

raw_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
vif_raw = compute_vif(df[raw_features])
print("\nRaw Features VIF Table (Severe Multicollinearity Observed):")
print(vif_raw.to_string(index=False))

# ---------------------------------------------------------
# Step 6: Dataset Splitting & Preprocessing
# ---------------------------------------------------------
print("\n[Step 6] Stratified/Partitioned 80/20 Train-Test Splitting...")
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training split: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
print(f"Testing split:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = ['ocean_proximity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + cat_names
print(f"Total processed feature dimension: {len(all_feature_names)} features")

# ---------------------------------------------------------
# Step 7: Model Training & Hyperparameter Tuning (GridSearchCV)
# ---------------------------------------------------------
print("\n[Step 7] Training OLS, Ridge (L2), and Lasso (L1) Models...")

# 1. Ordinary Least Squares (OLS)
print("Fitting Ordinary Least Squares (OLS) Baseline...")
ols_model = LinearRegression()
ols_model.fit(X_train_proc, y_train)

# 2. Ridge Regression with GridSearchCV
print("Tuning Ridge (L2) with 5-Fold GridSearchCV across alpha spectrum...")
ridge_alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
ridge_cv = GridSearchCV(
    Ridge(),
    param_grid={'alpha': ridge_alphas},
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
ridge_cv.fit(X_train_proc, y_train)
best_ridge = ridge_cv.best_estimator_
print(f"Optimal Ridge alpha: {ridge_cv.best_params_['alpha']} (CV RMSE: {-ridge_cv.best_score_:.2f})")

# 3. Lasso Regression with GridSearchCV
print("Tuning Lasso (L1) with 5-Fold GridSearchCV across alpha spectrum...")
lasso_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
lasso_cv = GridSearchCV(
    Lasso(tol=1e-2, max_iter=5000, random_state=42),
    param_grid={'alpha': lasso_alphas},
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
lasso_cv.fit(X_train_proc, y_train)
best_lasso = lasso_cv.best_estimator_
print(f"Optimal Lasso alpha: {lasso_cv.best_params_['alpha']} (CV RMSE: {-lasso_cv.best_score_:.2f})")

# ---------------------------------------------------------
# Step 8: Benchmarking & Performance Metrics
# ---------------------------------------------------------
print("\n[Step 8] Benchmarking Performance (Train/Test RMSE, MAE, R2)...")

def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    
    rmse_tr = np.sqrt(mean_squared_error(y_tr, pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    
    mae_tr = mean_absolute_error(y_tr, pred_tr)
    mae_te = mean_absolute_error(y_te, pred_te)
    
    r2_tr = r2_score(y_tr, pred_tr)
    r2_te = r2_score(y_te, pred_te)
    
    # 5-Fold Cross Validation on Train Set
    cv_scores = -cross_val_score(model, X_tr, y_tr, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    
    return {
        'Model': name,
        'Train RMSE ($)': rmse_tr,
        'Test RMSE ($)': rmse_te,
        'Train MAE ($)': mae_tr,
        'Test MAE ($)': mae_te,
        'Train R²': r2_tr,
        'Test R²': r2_te,
        '5-Fold CV RMSE ($)': cv_scores.mean(),
        'CV RMSE Std ($)': cv_scores.std()
    }

metrics = []
metrics.append(evaluate_model(ols_model, X_train_proc, y_train, X_test_proc, y_test, 'OLS Regression'))
metrics.append(evaluate_model(best_ridge, X_train_proc, y_train, X_test_proc, y_test, f"Ridge Regression (α={ridge_cv.best_params_['alpha']})"))
metrics.append(evaluate_model(best_lasso, X_train_proc, y_train, X_test_proc, y_test, f"Lasso Regression (α={lasso_cv.best_params_['alpha']})"))

metrics_df = pd.DataFrame(metrics)
print("\n" + "=" * 90)
print("COMPREHENSIVE REGRESSION BENCHMARK COMPARISON TABLE")
print("=" * 90)
print(metrics_df.to_string(index=False))

# Predictions for plotting
y_pred_ols = ols_model.predict(X_test_proc)
y_pred_ridge = best_ridge.predict(X_test_proc)
y_pred_lasso = best_lasso.predict(X_test_proc)

# ---------------------------------------------------------
# Step 9: Diagnostic Visualizations
# ---------------------------------------------------------
print("\n[Step 9] Generating Diagnostic Visualizations...")

# 1. Alpha Tuning Curves (Hyperparameter Search Path)
plt.figure(figsize=(14, 5))

# Ridge Curve
plt.subplot(1, 2, 1)
ridge_results = pd.DataFrame(ridge_cv.cv_results_)
plt.plot(ridge_results['param_alpha'].astype(float), -ridge_results['mean_test_score'], marker='o', color='#1f77b4', lw=2)
plt.xscale('log')
plt.axvline(ridge_cv.best_params_['alpha'], color='crimson', linestyle='--', label=f"Best α = {ridge_cv.best_params_['alpha']}")
plt.title('Ridge Regression: 5-Fold CV RMSE vs Alpha', fontsize=12, weight='bold')
plt.xlabel('Alpha Regularization Parameter (log scale)', fontsize=10)
plt.ylabel('Mean 5-Fold CV RMSE ($)', fontsize=10)
plt.legend()

# Lasso Curve
plt.subplot(1, 2, 2)
lasso_results = pd.DataFrame(lasso_cv.cv_results_)
plt.plot(lasso_results['param_alpha'].astype(float), -lasso_results['mean_test_score'], marker='s', color='#ff7f0e', lw=2)
plt.xscale('log')
plt.axvline(lasso_cv.best_params_['alpha'], color='crimson', linestyle='--', label=f"Best α = {lasso_cv.best_params_['alpha']}")
plt.title('Lasso Regression: 5-Fold CV RMSE vs Alpha', fontsize=12, weight='bold')
plt.xlabel('Alpha Regularization Parameter (log scale)', fontsize=10)
plt.ylabel('Mean 5-Fold CV RMSE ($)', fontsize=10)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "alpha_tuning_curves.png"), dpi=200, bbox_inches='tight')
plt.close()

# 2. Predicted vs Actual Values (3-Panel Parity Plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = [
    ('OLS Regression', y_pred_ols, '#2ca02c'),
    (f"Ridge Regression (α={ridge_cv.best_params_['alpha']})", y_pred_ridge, '#1f77b4'),
    (f"Lasso Regression (α={lasso_cv.best_params_['alpha']})", y_pred_lasso, '#ff7f0e')
]

for ax, (title, y_pred, color) in zip(axes, models):
    ax.scatter(y_test, y_pred, alpha=0.25, s=15, color=color, edgecolors='none')
    # 45-degree parity line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction ($y = \hat{y}$)')
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Actual Median House Value ($)', fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel('Predicted Median House Value ($)', fontsize=10)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 520000)
    ax.set_ylim(-20000, 550000)

plt.suptitle('Predicted vs Actual Median House Values (Test Set)', y=1.03, fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=200, bbox_inches='tight')
plt.close()

# 3. Residual Error Diagnostics (Residuals vs Fitted & Distribution Curves)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (title, y_pred, color) in enumerate(models):
    residuals = y_test - y_pred
    
    # Top Row: Residuals vs Fitted
    ax_top = axes[0, i]
    ax_top.scatter(y_pred, residuals, alpha=0.25, s=15, color=color, edgecolors='none')
    ax_top.axhline(0, color='crimson', linestyle='--', lw=2)
    ax_top.set_title(f'{title}\nResiduals vs Fitted Values', fontsize=11, weight='bold')
    ax_top.set_xlabel('Predicted House Value ($)', fontsize=10)
    if i == 0:
        ax_top.set_ylabel('Residual Error ($y - \hat{y}$)', fontsize=10)
    
    # Bottom Row: Residual Distribution
    ax_bot = axes[1, i]
    sns.histplot(residuals, kde=True, ax=ax_bot, color=color, bins=40, stat='density')
    ax_bot.axvline(0, color='crimson', linestyle='--', lw=1.5)
    ax_bot.set_title(f'{title}\nResidual Error Distribution', fontsize=11, weight='bold')
    ax_bot.set_xlabel('Residual Error ($y - \hat{y}$)', fontsize=10)
    if i == 0:
        ax_bot.set_ylabel('Density', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_analysis.png"), dpi=200, bbox_inches='tight')
plt.close()

# 4. Feature Coefficient & Shrinkage Comparison
coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'OLS': ols_model.coef_,
    'Ridge': best_ridge.coef_,
    'Lasso': best_lasso.coef_
})

plt.figure(figsize=(14, 8))
y_pos = np.arange(len(all_feature_names))
height = 0.25

plt.barh(y_pos - height, coef_df['OLS'], height=height, label='OLS', color='#2ca02c', alpha=0.85)
plt.barh(y_pos, coef_df['Ridge'], height=height, label=f"Ridge (α={ridge_cv.best_params_['alpha']})", color='#1f77b4', alpha=0.85)
plt.barh(y_pos + height, coef_df['Lasso'], height=height, label=f"Lasso (α={lasso_cv.best_params_['alpha']})", color='#ff7f0e', alpha=0.85)

plt.axvline(0, color='black', lw=1, linestyle='--')
plt.yticks(y_pos, all_feature_names, fontsize=10)
plt.xlabel('Standardized Coefficient Weight (Normalized Scale)', fontsize=11, weight='bold')
plt.title('Regularization Shrinkage Comparison: OLS vs Ridge vs Lasso Feature Weights', fontsize=13, weight='bold', pad=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "coefficient_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

print("\nCoefficient Summary Table (Standardized Predictors):")
print(coef_df.round(2).to_string(index=False))

print("\n" + "=" * 80)
print("TASK 2 EXECUTION COMPLETE: ALL VISUALIZATIONS EXPORTED TO 'visualizations/'")
print("=" * 80)
