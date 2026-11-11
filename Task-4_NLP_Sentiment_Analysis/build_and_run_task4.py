#!/usr/bin/env python3
"""
================================================================================
Datheon Machine Learning & Data Science Internship
Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)
================================================================================
Author: Datheon ML Intern
Objective:
  Preprocess raw textual corpus (HTML stripping, tokenization, stopword removal,
  lemmatization), extract TF-IDF embeddings (max_features=5000, n-grams=(1,2)),
  train and benchmark Multinomial Naive Bayes and Logistic Regression classifiers,
  and build an interactive predict_sentiment() helper for custom user inputs.
================================================================================
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# Styling aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "customer_reviews_corpus.csv")
STARTER_PATH = os.path.join(SCRIPT_DIR, "starter_reviews.csv")
VIZ_DIR = os.path.join(SCRIPT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Initialize NLP components
try:
    stop_words_set = set(stopwords.words('english'))
    # Preserve key negation markers that determine sentiment polarity
    negation_words = {'not', 'no', 'nor', 'neither', 'barely', 'hardly', 'scarcely', 'rarely', 'never'}
    stop_words_set = stop_words_set - negation_words
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"[!] NLTK resource error: {e}. Downloading missing datasets...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


def clean_and_lemmatize(text: str) -> str:
    """
    Production NLP cleaning pipeline:
    1. Strip HTML tags and markup.
    2. Strip URLs and hyperlinks.
    3. Remove non-alphabetic symbols (retaining alphabetic tokens).
    4. Convert to lower case.
    5. Tokenize and remove stopwords while preserving negation tokens.
    6. Lemmatize tokens to base dictionary forms (verbs and nouns).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Strip URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # 3. Strip non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 4. Lowercase
    tokens = text.lower().split()
    # 5 & 6. Filter stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos='v'), pos='n')
        for token in tokens
        if token not in stop_words_set and len(token) > 1
    ]
    return " ".join(cleaned_tokens)


def load_and_preprocess_data(corpus_path: str):
    """
    Load reviews corpus, filter to binary polarity (positive vs negative),
    apply text cleaning, and compute length diagnostics.
    """
    print(f"[*] Loading corpus from: {corpus_path}")
    df = pd.read_csv(corpus_path)
    print(f"    Raw records: {len(df)}")
    
    # Filter to binary sentiment for primary classification (positive vs negative)
    df_binary = df[df['sentiment'].isin(['positive', 'negative'])].copy()
    print(f"    Binary sentiment records: {len(df_binary)} (Filtered out neutral: {len(df) - len(df_binary)})")
    
    # Map target polarity: positive -> 1, negative -> 0
    df_binary['target'] = df_binary['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Compute raw text lengths
    df_binary['char_length'] = df_binary['review_text'].str.len()
    df_binary['word_count'] = df_binary['review_text'].str.split().str.len()

    print("[*] Preprocessing and lemmatizing text corpus...")
    df_binary['cleaned_text'] = df_binary['review_text'].apply(clean_and_lemmatize)
    df_binary['cleaned_word_count'] = df_binary['cleaned_text'].str.split().str.len()

    print(f"    Sample Raw:     \"{df_binary['review_text'].iloc[0]}\"")
    print(f"    Sample Cleaned: \"{df_binary['cleaned_text'].iloc[0]}\"")
    
    return df_binary


def plot_eda_distributions(df):
    """Plot target distribution and text length histograms."""
    print("[*] Generating EDA and Text Length Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Customer Reviews Corpus - Exploratory NLP Analysis", fontsize=15, fontweight='bold')

    # Subplot 1: Target class breakdown
    counts = df['sentiment'].value_counts()
    colors = ['#2b5c8f', '#d95f02']
    bars = axes[0].bar(counts.index.str.capitalize(), counts.values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        pct = (h / len(df)) * 100
        axes[0].text(b.get_x() + b.get_width()/2., h + 15, f"{h:,}\n({pct:.1f}%)", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_title("A. Sentiment Polarity Breakdown", fontweight='bold', fontsize=12)
    axes[0].set_ylabel("Review Count", fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.18)

    # Subplot 2: Character length KDE
    sns.kdeplot(data=df[df['sentiment']=='positive']['char_length'], ax=axes[1], label='Positive', color='#2b5c8f', fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df['sentiment']=='negative']['char_length'], ax=axes[1], label='Negative', color='#d95f02', fill=True, alpha=0.4, linewidth=2)
    axes[1].set_title("B. Character Length Distribution by Sentiment", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Review Character Count", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].legend(frameon=True)

    # Subplot 3: Cleaned token count boxplot
    sns.boxplot(x='sentiment', y='cleaned_word_count', data=df, ax=axes[2], palette=['#2b5c8f', '#d95f02'])
    axes[2].set_title("C. Preprocessed Word Count by Polarity", fontweight='bold', fontsize=12)
    axes[2].set_xlabel("Sentiment", fontsize=11)
    axes[2].set_ylabel("Cleaned Token Count", fontsize=11)
    axes[2].set_xticklabels(['Positive', 'Negative'])

    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "sentiment_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def build_and_train_pipeline(df):
    """
    Partition data into 80/20 stratified train/test sets,
    vectorize via TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
    and train MultinomialNB and LogisticRegression.
    """
    print("[*] Partitioning Dataset & Extracting TF-IDF Features...")
    X_raw = df['cleaned_text']
    y = df['target']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )

    # Fit strictly on training reviews
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"    TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"    Vocabulary Size: {vocab_size} n-grams extracted")

    # Define models
    nb_model = MultinomialNB(alpha=1.0)
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    models = {
        "Multinomial Naive Bayes": nb_model,
        "Logistic Regression": lr_model
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("[*] Benchmarking Models via 5-Fold Stratified Cross-Validation...")
    for name, model in models.items():
        # Cross validation on training TF-IDF
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_auc = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        # Fit model on entire training set
        model.fit(X_train_tfidf, y_train)

        # Inference on unseen test partition
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "cv_acc_mean": cv_scores.mean(),
            "cv_acc_std": cv_scores.std(),
            "cv_auc_mean": cv_auc.mean(),
            "test_acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "test_auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "prec_curve": precision_curve,
            "rec_curve": recall_curve,
            "avg_precision": ap,
            "cm": cm,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

        print(f"    --> {name}:")
        print(f"        5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"        5-Fold CV ROC-AUC:  {cv_auc.mean():.4f}")
        print(f"        Test Set Accuracy:  {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    return vectorizer, results, X_test_tfidf, y_test


def plot_confusion_matrices(results):
    """Plot side-by-side Confusion Matrices."""
    print("[*] Generating Confusion Matrices visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix Diagnostics on Unseen Test Reviews (N=242)", fontsize=15, fontweight='bold', y=0.98)

    models_info = [
        ("Multinomial Naive Bayes", axes[0], "Blues"),
        ("Logistic Regression", axes[1], "Greens")
    ]

    for name, ax, cmap in models_info:
        cm = results[name]['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

        sns.heatmap(cm, annot=annot_matrix, fmt="", cmap=cmap, cbar=False, ax=ax,
                    linewidths=1, linecolor='black', annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{name}\nAccuracy: {results[name]['test_acc']:.2%} | F1: {results[name]['f1']:.4f}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Sentiment", fontsize=11)
        ax.set_ylabel("True Sentiment", fontsize=11)
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(VIZ_DIR, "confusion_matrices.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_roc_pr_curves(results):
    """Plot ROC and Precision-Recall Curves."""
    print("[*] Generating ROC and Precision-Recall Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC and Precision-Recall Curves - NLP Sentiment Benchmark", fontsize=15, fontweight='bold')

    colors = {
        "Multinomial Naive Bayes": "#2b5c8f",
        "Logistic Regression": "#2ca02c"
    }

    # Subplot 1: ROC Curve
    for name, res in results.items():
        axes[0].plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['test_auc']:.4f})",
                     color=colors[name], linewidth=2.4)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (AUC = 0.5000)")
    axes[0].set_title("A. Receiver Operating Characteristic (ROC)", fontweight='bold', fontsize=12)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    # Subplot 2: Precision-Recall Curve
    for name, res in results.items():
        axes[1].plot(res['rec_curve'], res['prec_curve'], label=f"{name} (Avg Prec = {res['avg_precision']:.4f})",
                     color=colors[name], linewidth=2.4)
    axes[1].set_title("B. Precision-Recall Curve", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "roc_pr_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_top_features(vectorizer, results):
    """Extract and plot top 15 most positive and negative n-grams."""
    print("[*] Extracting Top TF-IDF Features...")
    feature_names = np.array(vectorizer.get_feature_names_out())
    lr_model = results["Logistic Regression"]['model']
    coefficients = lr_model.coef_[0]

    # Top positive and negative indices
    top_pos_idx = np.argsort(coefficients)[-12:]
    top_neg_idx = np.argsort(coefficients)[:12]

    top_pos_words = feature_names[top_pos_idx]
    top_pos_coefs = coefficients[top_pos_idx]

    top_neg_words = feature_names[top_neg_idx]
    top_neg_coefs = coefficients[top_neg_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Top Informative TF-IDF N-grams (Logistic Regression Coefficients)", fontsize=15, fontweight='bold')

    # Negative features
    axes[0].barh(top_neg_words, top_neg_coefs, color='#d95f02', edgecolor='black', linewidth=0.8)
    axes[0].set_title("A. Top Negative Sentiment Drivers", fontweight='bold', fontsize=12)
    axes[0].set_xlabel("Log-Odds Coefficient (Negative Influence)", fontsize=11)
    for i, v in enumerate(top_neg_coefs):
        axes[0].text(v - 0.08, i, f"{v:.2f}", va='center', ha='right', fontsize=9, fontweight='bold')

    # Positive features
    axes[1].barh(top_pos_words, top_pos_coefs, color='#2b5c8f', edgecolor='black', linewidth=0.8)
    axes[1].set_title("B. Top Positive Sentiment Drivers", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Log-Odds Coefficient (Positive Influence)", fontsize=11)
    for i, v in enumerate(top_pos_coefs):
        axes[1].text(v + 0.08, i, f"{v:.2f}", va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "top_tfidf_features.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def create_predict_sentiment_helper(vectorizer, results):
    """
    Construct user-facing predict_sentiment helper function.
    Tests custom phrases and starter reviews.
    """
    def predict_sentiment(text: str, model_name: str = "Logistic Regression") -> dict:
        """
        Predict sentiment polarity of arbitrary input text.
        Returns sentiment label, confidence, probability distribution, and cleaned tokens.
        """
        cleaned = clean_and_lemmatize(text)
        tfidf_vec = vectorizer.transform([cleaned])
        model = results[model_name]['model']

        pred_class = model.predict(tfidf_vec)[0]
        prob_dist = model.predict_proba(tfidf_vec)[0]

        sentiment = "Positive" if pred_class == 1 else "Negative"
        confidence = prob_dist[pred_class] * 100

        return {
            "input_text": text,
            "cleaned_tokens": cleaned,
            "sentiment": sentiment,
            "confidence": confidence,
            "prob_negative": prob_dist[0],
            "prob_positive": prob_dist[1],
            "model_used": model_name
        }

    return predict_sentiment


def test_starter_and_custom_inputs(predict_fn):
    """
    Evaluate predict_sentiment on user starter reviews and edge cases.
    Generate visual verification dashboard.
    """
    print("[*] Testing interactive predict_sentiment() function...")
    
    test_cases = [
        # Starter examples from user upload
        "The product quality is exceptional, exceeded all my expectations!",
        "Terrible customer service and the item arrived completely damaged.",
        "Absolutely love this! Easy to setup, robust, and works flawlessly.",
        "Stopped working after two days. Extremely disappointed with the purchase.",
        "Customer support helped resolve my setup question within ten minutes.",
        "Delivery was slow and package was missing one adapter cable.",
        "Clean interface and intuitive controls. Highly recommended for students.",
        "Not worth the money, plenty of better alternatives available.",
        "Great value and durable build. Would definitely purchase again.",
        # Live custom edge cases
        "This is an absolute masterpiece, highly recommend to anyone looking for quality!",
        "Defective out of the box, broken charger, and rude support agents.",
        "Not good at all, do not recommend buying this product."
    ]

    predictions = []
    print("\n" + "="*85)
    print("                 LIVE CUSTOM INPUT SENTIMENT TEST HELPER")
    print("="*85)
    for text in test_cases:
        res = predict_fn(text, model_name="Logistic Regression")
        icon = "[+]" if res['sentiment'] == "Positive" else "[-]"
        print(f"{icon} Text:       \"{text}\"")
        print(f"    Sentiment:  {res['sentiment']} (Confidence: {res['confidence']:.2f}%)")
        print(f"    Cleaned:    \"{res['cleaned_tokens']}\"")
        print(f"    P(Pos)={res['prob_positive']:.3f} | P(Neg)={res['prob_negative']:.3f}")
        print("-" * 85)
        predictions.append(res)

    # Plot visual dashboard of sample predictions
    fig, ax = plt.subplots(figsize=(12, 7))
    sample_texts = [f"#{i+1}: " + p['input_text'][:45] + ("..." if len(p['input_text']) > 45 else "") for i, p in enumerate(predictions)]
    probs = [p['prob_positive'] * 100 for p in predictions]
    colors = ['#2b5c8f' if p['sentiment'] == 'Positive' else '#d95f02' for p in predictions]

    bars = ax.barh(sample_texts[::-1], probs[::-1], color=colors[::-1], edgecolor='black', linewidth=0.8)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.7, label='Decision Threshold (50%)')
    ax.set_xlim(0, 105)
    ax.set_xlabel("Positive Sentiment Probability (%)", fontsize=11, fontweight='bold')
    ax.set_title("Interactive Sentiment Prediction Dashboard on Test Phrases", fontsize=14, fontweight='bold')

    for bar, p in zip(bars, predictions[::-1]):
        w = bar.get_width()
        lbl = f"{p['sentiment']} ({p['confidence']:.1f}%)"
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2., lbl, va='center', fontsize=9, fontweight='bold',
                color='#2b5c8f' if p['sentiment'] == 'Positive' else '#d95f02')

    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "interactive_prediction_dashboard.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def print_summary_table(results):
    """Print final benchmarking table."""
    print("\n" + "="*85)
    print("                 DATHEON TASK 4 MODEL PERFORMANCE BENCHMARK")
    print("="*85)
    header = f"{'Model Architecture':<26} | {'5-Fold CV Acc':<14} | {'Test Acc':<10} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 85)
    for name, res in results.items():
        row = f"{name:<26} | {res['cv_acc_mean']:.4f} (+/-{res['cv_acc_std']:.3f}) | {res['test_acc']:.4f}     | {res['recall']:.4f}   | {res['f1']:.4f}   | {res['test_auc']:.4f}"
        print(row)
    print("="*85 + "\n")


def main():
    df = load_and_preprocess_data(DATA_PATH)
    plot_eda_distributions(df)
    vectorizer, results, X_test_tfidf, y_test = build_and_train_pipeline(df)
    plot_confusion_matrices(results)
    plot_roc_pr_curves(results)
    plot_top_features(vectorizer, results)
    predict_fn = create_predict_sentiment_helper(vectorizer, results)
    test_starter_and_custom_inputs(predict_fn)
    print_summary_table(results)
    print("[SUCCESS] Task 4 Pipeline completed successfully. All artifacts saved.")


if __name__ == "__main__":
    main()
