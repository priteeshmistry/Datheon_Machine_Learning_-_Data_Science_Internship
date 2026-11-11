#!/usr/bin/env python3
"""
================================================================================
Datheon Machine Learning & Data Science Internship
Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)
================================================================================
Author: Datheon ML Intern
Objective:
  Build, tune, and evaluate an ensemble classification pipeline to predict customer
  churn using the Telco Customer Churn dataset. Address severe class imbalance using
  SMOTE and cost-sensitive class weighting, evaluate cross-validated ROC-AUC (>80%),
  and extract the top influential feature importances.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
    log_loss
)

# Set visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(SCRIPT_DIR, "telco_churn.csv")

VIZ_DIR = os.path.join(SCRIPT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def check_and_import_dependencies():
    """Verify if XGBoost and Imbalanced-Learn are installed."""
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except ImportError:
        print("[!] XGBoost not found. Please install xgboost.")
        sys.exit(1)

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[!] imbalanced-learn not found. Please install imbalanced-learn.")
        sys.exit(1)

    return XGBClassifier, SMOTE


def load_and_clean_data(file_path):
    """
    Load Telco Churn dataset and perform initial hygiene:
    1. Strip customerID.
    2. Convert TotalCharges from object/string to float, imputing 11 tenure=0 rows to 0.0.
    3. Binary encode target 'Churn' (Yes: 1, No: 0).
    """
    print(f"[*] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"    Raw Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. TotalCharges conversion and handling whitespace
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    null_total = df['TotalCharges'].isnull().sum()
    print(f"    Whitespace/Null values found in TotalCharges: {null_total}")
    # Customers with tenure == 0 have not been billed yet; impute with 0.0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # 2. Encode target variable
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"    Target Distribution:")
    print(f"      No Churn (0): {churn_counts['No']} ({churn_pct['No']:.2f}%)")
    print(f"      Churn (1):    {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
    print(f"      Imbalance Ratio: {churn_counts['No']/churn_counts['Yes']:.2f}:1")

    return df


def generate_eda_plots(df):
    """Generate exploratory data analysis visualizations."""
    print("[*] Generating Exploratory Data Analysis visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Telco Customer Churn - Exploratory Data Analysis", fontsize=16, fontweight='bold', y=0.98)

    # Subplot 1: Target Class Distribution
    ax1 = axes[0, 0]
    churn_counts = df['Churn'].value_counts()
    colors = ['#2b5c8f', '#d95f02']
    bars = ax1.bar(['No Churn (0)', 'Churn (1)'], churn_counts.values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., h + 60, f'{h:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_title("A. Target Class Distribution (Imbalance: ~73.4% vs 26.6%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Customer Count", fontsize=11)
    ax1.set_ylim(0, max(churn_counts.values) * 1.18)

    # Subplot 2: Churn by Contract Type
    ax2 = axes[0, 1]
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn[['No', 'Yes']].plot(kind='bar', stacked=True, ax=ax2, color=['#2b5c8f', '#d95f02'], edgecolor='black', linewidth=0.8)
    ax2.set_title("B. Churn Proportion by Contract Type", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_xlabel("Contract Type", fontsize=11)
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(["No Churn", "Churn"], loc='upper right', frameon=True)
    for i, (contract, row) in enumerate(contract_churn.iterrows()):
        ax2.text(i, row['No']/2, f"{row['No']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax2.text(i, row['No'] + row['Yes']/2, f"{row['Yes']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Subplot 3: Churn by Internet Service Type
    ax3 = axes[1, 0]
    internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    internet_churn[['No', 'Yes']].plot(kind='bar', stacked=True, ax=ax3, color=['#2b5c8f', '#d95f02'], edgecolor='black', linewidth=0.8)
    ax3.set_title("C. Churn Proportion by Internet Service", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Percentage (%)", fontsize=11)
    ax3.set_xlabel("Internet Service Type", fontsize=11)
    ax3.tick_params(axis='x', rotation=0)
    ax3.legend(["No Churn", "Churn"], loc='upper right', frameon=True)
    for i, (service, row) in enumerate(internet_churn.iterrows()):
        ax3.text(i, row['No']/2, f"{row['No']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax3.text(i, row['No'] + row['Yes']/2, f"{row['Yes']:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Subplot 4: Tenure Distribution by Churn Status (KDE)
    ax4 = axes[1, 1]
    sns.kdeplot(data=df[df['Churn'] == 'No']['tenure'], ax=ax4, label='No Churn (Retained)', color='#2b5c8f', fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df['Churn'] == 'Yes']['tenure'], ax=ax4, label='Churn (Attrited)', color='#d95f02', fill=True, alpha=0.4, linewidth=2)
    ax4.set_title("D. Customer Tenure Density by Churn Status", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Tenure (Months with Company)", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.legend(loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(VIZ_DIR, "churn_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def prepare_features_and_split(df):
    """
    Split into features X and target y.
    Construct column transformer for continuous scaling and one-hot categorical encoding.
    """
    drop_cols = ['customerID', 'Churn', 'Churn_Numeric']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['Churn_Numeric'].copy()

    # Identify numerical and categorical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    print(f"[*] Features: {len(numeric_cols)} Numerical, {len(categorical_cols)} Categorical")

    # Stratified Train-Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"    Train Churn Rate: {y_train.mean():.4f} | Test Churn Rate: {y_test.mean():.4f}")

    # Build preprocessing column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    
    # Extract feature names after encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"    Total Encoded Features: {len(all_feature_names)}")
    return (X_train_proc, X_test_proc, y_train, y_test, all_feature_names, preprocessor)


def apply_smote_resampling(X_train, y_train, SMOTE_cls):
    """
    Apply SMOTE to synthesize minority class samples exclusively on training set.
    """
    print("[*] Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
    before_counts = np.bincount(y_train)
    smote = SMOTE_cls(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    after_counts = np.bincount(y_resampled)
    
    print(f"    Before SMOTE: Class 0 = {before_counts[0]}, Class 1 = {before_counts[1]} (Total: {len(y_train)})")
    print(f"    After SMOTE:  Class 0 = {after_counts[0]}, Class 1 = {after_counts[1]} (Total: {len(y_resampled)})")

    # Plot SMOTE before and after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Imbalance Resolution: SMOTE Resampling Analysis", fontsize=15, fontweight='bold')

    colors = ['#2b5c8f', '#d95f02']
    # Before
    bars1 = axes[0].bar(['No Churn (0)', 'Churn (1)'], before_counts, color=colors, width=0.5, edgecolor='black')
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2., b.get_height() + 50, f"{b.get_height():,}", ha='center', fontweight='bold')
    axes[0].set_title(f"A. Training Distribution Before SMOTE (N={len(y_train):,})", fontweight='bold')
    axes[0].set_ylabel("Sample Count")
    axes[0].set_ylim(0, max(after_counts) * 1.18)

    # After
    bars2 = axes[1].bar(['No Churn (0)', 'Churn (1)'], after_counts, color=colors, width=0.5, edgecolor='black')
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2., b.get_height() + 50, f"{b.get_height():,}", ha='center', fontweight='bold')
    axes[1].set_title(f"B. Training Distribution After SMOTE (N={len(y_resampled):,})", fontweight='bold')
    axes[1].set_ylabel("Sample Count")
    axes[1].set_ylim(0, max(after_counts) * 1.18)

    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "smote_resampling_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")

    return X_resampled, y_resampled


def train_and_evaluate_models(X_train_res, y_train_res, X_train_orig, y_train_orig, X_test, y_test, XGBClassifier_cls):
    """
    Train Random Forest and XGBoost with Stratified K-Fold Cross Validation.
    Evaluate both SMOTE and Class-Weighting paradigms.
    """
    print("[*] Training and Evaluating Models...")

    # Calculate scale_pos_weight for XGBoost
    neg_count = np.sum(y_train_orig == 0)
    pos_count = np.sum(y_train_orig == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"    Computed XGBoost scale_pos_weight: {scale_pos_weight:.3f}")

    # Initialize models
    # 1. Random Forest (Trained on SMOTE resampled data)
    rf_smote = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    # 2. Random Forest (Class Weighting on original data)
    rf_weighted = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 3. XGBoost (Trained on SMOTE resampled data)
    xgb_smote = XGBClassifier_cls(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    # 4. XGBoost (Trained with scale_pos_weight on original data)
    xgb_weighted = XGBClassifier_cls(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    models = {
        "Random Forest (SMOTE)": (rf_smote, X_train_res, y_train_res),
        "Random Forest (Weighted)": (rf_weighted, X_train_orig, y_train_orig),
        "XGBoost (SMOTE)": (xgb_smote, X_train_res, y_train_res),
        "XGBoost (Weighted)": (xgb_weighted, X_train_orig, y_train_orig)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, X_tr, y_tr) in models.items():
        print(f"    --> Fitting {name}...")
        # 5-fold cross-validation on the model's training data
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Fit on entire respective training split
        model.fit(X_tr, y_tr)

        # Predict probabilities on unseen test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_prob)
        test_acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        loss = log_loss(y_test, y_prob)

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "test_auc": test_auc,
            "test_acc": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": loss,
            "fpr": fpr,
            "tpr": tpr,
            "cm": cm,
            "y_prob": y_prob,
            "y_pred": y_pred
        }

        print(f"        5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"        Test Set ROC-AUC:  {test_auc:.4f} [Requirement >80%: {'PASS' if test_auc > 0.80 else 'FAIL'}]")
        print(f"        Test Set Recall:   {recall:.4f} | F1-Score: {f1:.4f} | Accuracy: {test_acc:.4f}")

    return results


def plot_roc_curves(results):
    """Plot comprehensive ROC-AUC comparison curves."""
    print("[*] Generating ROC-AUC Curve visualization...")
    plt.figure(figsize=(9, 7))

    colors = {
        "Random Forest (SMOTE)": "#2b5c8f",
        "Random Forest (Weighted)": "#41b6c4",
        "XGBoost (SMOTE)": "#d95f02",
        "XGBoost (Weighted)": "#7570b3"
    }
    linestyles = {
        "Random Forest (SMOTE)": "-",
        "Random Forest (Weighted)": "--",
        "XGBoost (SMOTE)": "-",
        "XGBoost (Weighted)": "-."
    }

    for name, res in results.items():
        plt.plot(
            res['fpr'],
            res['tpr'],
            label=f"{name} (AUC = {res['test_auc']:.4f})",
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2.2
        )

    # Reference diagonal (Random Guess)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (AUC = 0.5000)", linewidth=1.2)

    # Benchmark threshold line (80% AUC)
    plt.axhline(0.80, color='gray', linestyle=':', alpha=0.6)
    plt.text(0.60, 0.81, 'Datheon Benchmark Requirement: AUC > 0.80', fontsize=10, color='#333333', style='italic')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight='bold')
    plt.title("Receiver Operating Characteristic (ROC) Benchmark Curves\nPredictive Customer Churn Classifier", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(VIZ_DIR, "roc_auc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_confusion_matrices(results, y_test):
    """Plot side-by-side confusion matrices for Random Forest and XGBoost."""
    print("[*] Generating Confusion Matrices visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Confusion Matrix Diagnostics on Unseen Test Set (N=1,409)", fontsize=15, fontweight='bold', y=0.98)

    models_to_plot = [
        ("Random Forest (SMOTE)", axes[0, 0]),
        ("Random Forest (Weighted)", axes[0, 1]),
        ("XGBoost (SMOTE)", axes[1, 0]),
        ("XGBoost (Weighted)", axes[1, 1])
    ]

    for name, ax in models_to_plot:
        cm = results[name]['cm']
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues" if "Random Forest" in name else "Oranges",
            cbar=False,
            ax=ax,
            linewidths=1,
            linecolor='black',
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"{name}\nRecall: {results[name]['recall']:.2%} | AUC: {results[name]['test_auc']:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticklabels(['Retained (0)', 'Churned (1)'])
        ax.set_yticklabels(['Retained (0)', 'Churned (1)'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(VIZ_DIR, "confusion_matrices.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_feature_importances(results, feature_names):
    """Extract and plot top 15 most influential features for RF and XGBoost."""
    print("[*] Extracting and Plotting Feature Importances...")
    rf_model = results["Random Forest (SMOTE)"]['model']
    xgb_model = results["XGBoost (Weighted)"]['model']

    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Top Influential Drivers of Customer Churn", fontsize=15, fontweight='bold')

    # Subplot 1: Random Forest
    axes[0].barh(rf_imp.index[::-1], rf_imp.values[::-1], color='#2b5c8f', edgecolor='black', linewidth=0.8)
    axes[0].set_title("A. Random Forest (Mean Gini Impurity Decrease)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Relative Importance Score", fontsize=11)
    for i, v in enumerate(rf_imp.values[::-1]):
        axes[0].text(v + 0.003, i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    # Subplot 2: XGBoost
    axes[1].barh(xgb_imp.index[::-1], xgb_imp.values[::-1], color='#d95f02', edgecolor='black', linewidth=0.8)
    axes[1].set_title("B. XGBoost (Average Gain / Split Influence)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Relative Importance Score", fontsize=11)
    for i, v in enumerate(xgb_imp.values[::-1]):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def print_summary_table(results):
    """Display final benchmarking table."""
    print("\n" + "="*85)
    print("                 DATHEON TASK 3 MODEL PERFORMANCE BENCHMARK")
    print("="*85)
    header = f"{'Model':<28} | {'5-Fold CV AUC':<14} | {'Test AUC':<10} | {'Recall':<8} | {'F1':<8} | {'Accuracy':<8}"
    print(header)
    print("-" * 85)
    for name, res in results.items():
        row = f"{name:<28} | {res['cv_auc_mean']:.4f} (+/-{res['cv_auc_std']:.3f}) | {res['test_auc']:.4f}     | {res['recall']:.4f}   | {res['f1']:.4f}   | {res['test_acc']:.4f}"
        print(row)
    print("="*85)
    print(" Benchmark Acceptance Criterion: Test ROC-AUC > 0.80 (80.00%)")
    all_pass = all(res['test_auc'] > 0.80 for res in results.values())
    print(f" Status: {'ALL MODELS SATISFY CRITERIA (>80% ROC-AUC)' if all_pass else 'CRITERIA NOT MET'}")
    print("="*85 + "\n")


def main():
    XGBClassifier, SMOTE = check_and_import_dependencies()
    df = load_and_clean_data(DATA_PATH)
    generate_eda_plots(df)
    X_train_proc, X_test_proc, y_train, y_test, feature_names, preprocessor = prepare_features_and_split(df)
    X_train_res, y_train_res = apply_smote_resampling(X_train_proc, y_train, SMOTE)
    results = train_and_evaluate_models(X_train_res, y_train_res, X_train_proc, y_train, X_test_proc, y_test, XGBClassifier)
    plot_roc_curves(results)
    plot_confusion_matrices(results, y_test)
    plot_feature_importances(results, feature_names)
    print_summary_table(results)
    print("[SUCCESS] Pipeline completed successfully. All artifacts saved.")


if __name__ == "__main__":
    main()
