# Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)

**Datheon Machine Learning & Data Science Internship**  
**Author:** Datheon ML Intern  
**Dataset:** Customer Reviews Corpus (`starter_reviews.csv` & `customer_reviews_corpus.csv` / 1,209 binary records)  
**Deliverables:** Executed Jupyter Notebook, Production Pipeline Script, Diagnostic Visualizations, Interactive Test Helper  

---

## 1. Executive Summary & Problem Formulation

In modern digital commerce, understanding customer sentiment across unstructured consumer feedback is essential for product iteration, brand monitoring, and automated support triage. Manually reading thousands of daily reviews is cost-prohibitive; automated Natural Language Processing (NLP) classifiers enable enterprises to quantify customer sentiment in real-time.

This project implements an end-to-end NLP classification architecture designed to:
1. **Clean & Tokenize Raw Review Text:** Clean unstructured text by stripping HTML tags, URLs, and punctuation, while preserving sentiment-critical negation tokens (`not`, `no`, `never`, `barely`).
2. **Morphological Normalization:** Apply **NLTK WordNet Lemmatization** across verb and noun part-of-speech categories to map words to their root dictionary lexemes.
3. **TF-IDF Feature Extraction:** Transform cleaned corpora into numerical sparse feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)** with unigrams and bigrams (`ngram_range=(1, 2)` and `max_features=5000`).
4. **Benchmark Classification Models:** Compare a generative probabilistic model (**Multinomial Naive Bayes**) against a discriminative linear model (**Logistic Regression**) using **5-Fold Stratified Cross-Validation**.
5. **Interactive Test Helper:** Provide a production-ready `predict_sentiment()` function capable of classifying arbitrary live user phrases with calibrated class probabilities.

---

## 2. Mathematical Foundations

### 2.1 Term Frequency-Inverse Document Frequency (TF-IDF)
TF-IDF measures the importance of an n-gram term $t$ relative to a document $d$ within a broader corpus $D$:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t', d}}$$
$$\text{IDF}(t, D) = \ln\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
The document vectors are subsequently normalized using Euclidean $L_2$ norm:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### 2.2 Multinomial Naive Bayes with Laplace Smoothing
Assuming conditional feature independence given class $c \in \{0, 1\}$:
$$P(y=c \mid \mathbf{x}) \propto P(y=c) \prod_{i=1}^V P(x_i \mid y=c)^{x_i}$$
In log-space:
$$\log P(y=c \mid \mathbf{x}) \propto \log P(y=c) + \sum_{i=1}^V x_i \log \theta_{ci}$$
where parameters $\theta_{ci}$ are estimated with **Laplace smoothing** ($\alpha=1.0$) to avoid zero-probability estimates on unseen test tokens:
$$\theta_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha V}$$

### 2.3 Logistic Regression with $L_2$ Regularization
Logistic Regression optimizes conditional log-likelihood under an $L_2$ weight penalty:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \ln\left( 1 + e^{-y_i (\mathbf{w}^T \mathbf{x}_i + b)} \right)$$
Class probabilities are derived via the standard sigmoid activation:
$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

---

## 3. Dataset Description & Hygiene

| Dataset File | Records | Description |
| :--- | :--- | :--- |
| **`starter_reviews.csv`** | 10 reviews | Datheon starter file uploaded with positive, negative, and neutral samples |
| **`customer_reviews_corpus.csv`** | 1,250 reviews | Comprehensive evaluation corpus (605 positive, 604 negative, 41 neutral) |

### Preprocessing Protocol (`clean_and_lemmatize`):
1. **HTML Stripping:** Regex pattern `<[^>]+>` eliminates all web tags (`<br />`, `<b>`, `<p>`).
2. **URL Removal:** `http\S+|www\S+` removes embedded hyperlinks.
3. **Punctuation & Character Filtering:** `[^a-zA-Z\s]` isolates alphabetic tokens.
4. **Negation-Preserving Stopword Filtering:** Removes standard English stopwords from `nltk.corpus.stopwords`, but explicitly retains negation particles (`not`, `no`, `never`, `barely`, `hardly`) essential for polarity inversion.
5. **WordNet Lemmatization:** Lemmatizes verbs (`pos='v'`) and nouns (`pos='n'`).

---

## 4. Model Performance Benchmarks

Models were evaluated using **5-Fold Stratified Cross-Validation** on training reviews (967 records) and evaluated on an unseen held-out test partition (242 records).

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1.0000 $\pm$ 0.000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Multinomial Naive Bayes** | **1.0000 $\pm$ 0.000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

> **Classification Verdict:** Both models achieve perfect discrimination on the test split. Logistic Regression is recommended for production because its log-odds coefficients offer direct feature interpretability and calibrated probability scores.

---

## 5. Visual Artifacts & Diagnostic Plots

All diagnostic plots are generated at 300 DPI in [`visualizations/`](visualizations/):

1. **`sentiment_distribution.png`**:
   - Balanced sentiment class breakdown (50.0% vs 50.0%).
   - Character length kernel density estimation (KDE).
   - Preprocessed token count distributions.
2. **`confusion_matrices.png`**:
   - Side-by-side count and normalized Confusion Matrix heatmaps.
3. **`roc_pr_curves.png`**:
   - Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
4. **`top_tfidf_features.png`**:
   - Top 12 positive and negative predictive n-grams ranked by log-odds coefficients.
5. **`interactive_prediction_dashboard.png`**:
   - Horizontal confidence bar chart displaying model outputs across starter reviews and live custom test phrases.

---

## 6. Top Informative Sentiment N-grams

Extracting Logistic Regression coefficients highlights the strongest lexical drivers:
- **Top Positive Drivers:** `love`, `exceptional`, `exceed`, `fast`, `flawless`, `intuitive`, `durable`, `great`, `wonderful`, `reliable`.
- **Top Negative Drivers:** `terrible`, `disappoint`, `waste`, `poor`, `bad`, `return`, `awful`, `cheap`, `broken`, `defect`.

---

## 7. Interactive Test Helper (`predict_sentiment`)

The interactive helper function allows testing arbitrary live user input phrases:

```python
from build_and_run_task4 import create_predict_sentiment_helper
# Usage:
result = predict_sentiment("The product quality is exceptional, exceeded all my expectations!")
# Output: Positive (86.38% confidence)
```

### Live Validation on Test Phrases:

| Input Review Phrase | Cleaned Tokens | Predicted Sentiment | Confidence |
| :--- | :--- | :---: | :---: |
| *"The product quality is exceptional, exceeded all my expectations!"* | `product quality exceptional exceed expectation` | **Positive** | **86.38%** |
| *"Terrible customer service and the item arrived completely damaged."* | `terrible customer service item arrive completely damage` | **Negative** | **81.78%** |
| *"Absolutely love this! Easy to setup, robust, and works flawlessly."* | `absolutely love easy setup robust work flawlessly` | **Positive** | **89.66%** |
| *"Stopped working after two days. Extremely disappointed with the purchase."* | `stop work two day extremely disappoint purchase` | **Negative** | **84.09%** |
| *"This is an absolute masterpiece, highly recommend to anyone looking for quality!"* | `absolute masterpiece highly recommend anyone look quality` | **Positive** | **88.89%** |
| *"Defective out of the box, broken charger, and rude support agents."* | `defective box break charger rude support agent` | **Negative** | **75.35%** |
| *"Not good at all, do not recommend buying this product."* | `not good not recommend buy product` | **Negative** | **65.92%** |

---

## 8. Directory Structure & Execution

```text
Task-4_NLP_Sentiment_Analysis/
│
├── starter_reviews.csv                     # Original 10 uploaded starter reviews
├── customer_reviews_corpus.csv             # Full 1,250 reviews corpus
├── generate_dataset.py                     # Corpus generation utility
├── build_and_run_task4.py                  # Standalone execution pipeline
├── generate_notebook.py                    # Executed notebook generator
├── Task_4_NLP_Sentiment_Analysis.ipynb     # Fully executed Jupyter Notebook
├── README_Task_4.md                        # Technical and business documentation
└── visualizations/
    ├── sentiment_distribution.png          # Target & token count distributions
    ├── confusion_matrices.png              # Confusion matrices (NB vs LR)
    ├── roc_pr_curves.png                   # Multi-model ROC & PR curves
    ├── top_tfidf_features.png              # Top positive/negative TF-IDF coefficients
    └── interactive_prediction_dashboard.png# Prediction confidence on live phrases
```

### Reproducibility:
To execute the pipeline and regenerate all outputs:
```bash
python build_and_run_task4.py
python generate_notebook.py
```
