import os
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

WORKSPACE = r"C:\Users\Priteesh Mistry\My project"
CSV_PATH = os.path.join(WORKSPACE, "iris.csv")
OUTPUT_DIR = os.path.join(WORKSPACE, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Step 1: Loading Dataset ---")
df = pd.read_csv(CSV_PATH)
print(f"Dataset Shape: {df.shape}")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['species'].value_counts())

# 2. EDA Visualizations
print("\n--- Step 2: Generating EDA Plots ---")
# Pairplot
pairplot_fig = sns.pairplot(df, hue="species", diag_kind="kde", markers=["o", "s", "D"], palette="Set2")
pairplot_fig.fig.suptitle("Iris Feature Pairwise Distributions by Species", y=1.02, fontsize=14, weight='bold')
pairplot_path = os.path.join(OUTPUT_DIR, "pairplot.png")
pairplot_fig.savefig(pairplot_path, bbox_inches='tight', dpi=200)
plt.close()

# Correlation Matrix
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=['species'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', vmin=-1, vmax=1, mask=mask, square=True, linewidths=.5)
plt.title("Iris Feature Correlation Matrix", fontsize=14, weight='bold', pad=12)
corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
plt.savefig(corr_path, bbox_inches='tight', dpi=200)
plt.close()

# Feature Distributions Boxplot
plt.figure(figsize=(12, 6))
features = numeric_df.columns
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=df, palette='Set2')
    plt.title(f"Distribution of {feature.replace('_', ' ').title()}", fontsize=11, weight='bold')
    plt.xlabel("")
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "feature_distributions.png")
plt.savefig(dist_path, bbox_inches='tight', dpi=200)
plt.close()

# 3. Preprocessing
print("\n--- Step 3: Preprocessing & 80/20 Splitting ---")
X = df.drop(columns=['species'])
y = df['species']

# Label encoding for y
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)
print(f"Training set: {X_train.shape[0]} samples (80%)")
print(f"Testing set: {X_test.shape[0]} samples (20%)")

# Feature Scaling with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train features mean before scaling: {np.round(X_train.mean().values, 2)}")
print(f"Train features std before scaling:  {np.round(X_train.std().values, 2)}")
print(f"Train features mean after scaling:  {np.round(X_train_scaled.mean(axis=0), 2)}")
print(f"Train features std after scaling:   {np.round(X_train_scaled.std(axis=0), 2)}")

# 4. Model Training & Cross Validation
print("\n--- Step 4: Model Training ---")
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train_scaled, y_train)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train) # Decision tree trained on unscaled features (scale invariant)

# 5-fold Stratified Cross-Validation on training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"Logistic Regression 5-Fold CV Accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")
print(f"Decision Tree 5-Fold CV Accuracy:       {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})")

# 5. Model Evaluation
print("\n--- Step 5: Test Set Predictions & Benchmarking ---")
lr_pred = lr_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test)

# Metrics calculation
def calculate_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    prec_weighted = precision_score(y_true, y_pred, average='weighted')
    rec_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro,
        "Precision (Weighted)": prec_weighted,
        "Recall (Weighted)": rec_weighted,
        "F1-Score (Weighted)": f1_weighted
    }

benchmark_df = pd.DataFrame([
    calculate_metrics(y_test, lr_pred, "Logistic Regression"),
    calculate_metrics(y_test, dt_pred, "Decision Tree (max_depth=3)")
])
print("\n--- Benchmark Comparison Table ---")
print(benchmark_df.to_string(index=False))

print("\n--- Logistic Regression Classification Report ---")
lr_report = classification_report(y_test, lr_pred, target_names=class_names)
print(lr_report)

print("\n--- Decision Tree Classification Report ---")
dt_report = classification_report(y_test, dt_pred, target_names=class_names)
print(dt_report)

# 6. Confusion Matrix Heatmaps
print("\n--- Step 6: Plotting Confusion Matrices ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title(f"Logistic Regression Confusion Matrix\n(Accuracy: {accuracy_score(y_test, lr_pred)*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[0].set_xlabel("Predicted Label", fontsize=11, weight='bold')
axes[0].set_ylabel("True Label", fontsize=11, weight='bold')

cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title(f"Decision Tree Confusion Matrix\n(Accuracy: {accuracy_score(y_test, dt_pred)*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[1].set_xlabel("Predicted Label", fontsize=11, weight='bold')
axes[1].set_ylabel("True Label", fontsize=11, weight='bold')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, bbox_inches='tight', dpi=200)
plt.close()

# 7. Decision Boundary Visualization (Petal Length vs Petal Width)
print("\n--- Step 7: Plotting Decision Boundaries ---")
# We train 2D models on petal_length and petal_width for visualization
X_2d = df[['petal_length', 'petal_width']].values
y_2d = y_encoded

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.20, random_state=42, stratify=y_2d
)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

lr_2d = LogisticRegression(max_iter=200, random_state=42)
lr_2d.fit(X_train_2d_scaled, y_train_2d)

dt_2d = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_2d.fit(X_train_2d_scaled, y_train_2d)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Meshgrid
x_min, x_max = X_train_2d_scaled[:, 0].min() - 0.8, X_train_2d_scaled[:, 0].max() + 0.8
y_min, y_max = X_train_2d_scaled[:, 1].min() - 0.8, X_train_2d_scaled[:, 1].max() + 0.8
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

colors = ['#4A90E2', '#50E3C2', '#E74C3C']
cmap_light = ['#D4E6F1', '#D1F2EB', '#FADBD8']
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(cmap_light)
scatter_cmap = ListedColormap(colors)

# Logistic Regression Decision Boundary
Z_lr = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_lr, alpha=0.6, cmap=custom_cmap)
axes[0].contour(xx, yy, Z_lr, colors='k', linewidths=0.8, alpha=0.7)

scatter_train_0 = axes[0].scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], c=y_train_2d, cmap=scatter_cmap, edgecolors='k', s=50, alpha=0.8, label="Train")
scatter_test_0 = axes[0].scatter(X_test_2d_scaled[:, 0], X_test_2d_scaled[:, 1], c=y_test_2d, cmap=scatter_cmap, edgecolors='yellow', linewidth=1.5, s=80, marker='*', label="Test")
axes[0].set_title("Logistic Regression Decision Boundaries\n(Smooth Linear Boundaries)", fontsize=13, weight='bold')
axes[0].set_xlabel("Standardized Petal Length", fontsize=11)
axes[0].set_ylabel("Standardized Petal Width", fontsize=11)

# Decision Tree Decision Boundary
Z_dt = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_dt = Z_dt.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_dt, alpha=0.6, cmap=custom_cmap)
axes[1].contour(xx, yy, Z_dt, colors='k', linewidths=0.8, alpha=0.7)

scatter_train_1 = axes[1].scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], c=y_train_2d, cmap=scatter_cmap, edgecolors='k', s=50, alpha=0.8, label="Train")
scatter_test_1 = axes[1].scatter(X_test_2d_scaled[:, 0], X_test_2d_scaled[:, 1], c=y_test_2d, cmap=scatter_cmap, edgecolors='yellow', linewidth=1.5, s=80, marker='*', label="Test")
axes[1].set_title("Decision Tree Decision Boundaries\n(Orthogonal Axis-Aligned Splits)", fontsize=13, weight='bold')
axes[1].set_xlabel("Standardized Petal Length", fontsize=11)
axes[1].set_ylabel("Standardized Petal Width", fontsize=11)

# Legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Class: {class_names[0]}', markerfacecolor=colors[0], markersize=9),
    Line2D([0], [0], marker='o', color='w', label=f'Class: {class_names[1]}', markerfacecolor=colors[1], markersize=9),
    Line2D([0], [0], marker='o', color='w', label=f'Class: {class_names[2]}', markerfacecolor=colors[2], markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Train Sample (black border)', markerfacecolor='gray', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='*', color='w', label='Test Sample (yellow border)', markerfacecolor='gray', markeredgecolor='gold', markersize=12)
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=True)

plt.tight_layout()
db_path = os.path.join(OUTPUT_DIR, "decision_boundaries.png")
plt.savefig(db_path, bbox_inches='tight', dpi=200)
plt.close()

# 8. Decision Tree Structure & Feature Importance
print("\n--- Step 8: Plotting Tree Structure & Feature Importance ---")
plt.figure(figsize=(14, 8))
plot_tree(dt_model, feature_names=list(X.columns), class_names=class_names,
          filled=True, rounded=True, precision=2, fontsize=11)
plt.title("Trained Decision Tree Visualization (max_depth=3)", fontsize=15, weight='bold', pad=12)
tree_path = os.path.join(OUTPUT_DIR, "decision_tree_structure.png")
plt.savefig(tree_path, bbox_inches='tight', dpi=200)
plt.close()

# Feature Importances
plt.figure(figsize=(8, 4.5))
fi = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_fi = ['#85C1E9' if v < 0.1 else '#2E86C1' for v in fi.values]
fi.plot(kind='barh', color=colors_fi, edgecolor='black', linewidth=0.8)
plt.title("Decision Tree Feature Importance", fontsize=13, weight='bold', pad=10)
plt.xlabel("Importance Score (Gini Importance)", fontsize=11)
for i, v in enumerate(fi.values):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=10)
plt.xlim(0, max(fi.values) + 0.1)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path, bbox_inches='tight', dpi=200)
plt.close()

print("\nAll pipeline execution and visualizations completed successfully!")
