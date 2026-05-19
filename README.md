# A Tri-Ensemble Stacking Framework for E-Commerce Purchase Prediction with Explainable AI

Developed a two-layer **Stacking Ensemble** framework combining Random Forest, Extra Trees, and XGBoost to predict online buyer intent with **90.23% accuracy** over 12,330 session records. Handled severe class imbalances via **SMOTE** and deployed **SHAP** to map complex black-box metrics into transparent, actionable business insights.

---

## 📌 Project Overview
Online e-commerce consumer behavior data is inherently skewed, with only a small fraction of sessions resulting in a purchase (~15.5%). Standard machine learning models often struggle with this class imbalance, yielding high accuracy by simply predicting the majority class while completely missing actual buyers. Furthermore, high-performing ensemble models typically act as "black boxes," making it difficult for businesses to trust or utilize their predictions.

This project introduces a robust, end-to-end machine learning pipeline that addresses both challenges. By leveraging an optimized tri-ensemble stacking classifier alongside Explainable AI (XAI) frameworks, it accurately predicts purchasing intent and surfaces the underlying behavioral catalysts that drive conversions.

---

## 🛠️ Tech Stack & Core Techniques
* **Programming Language:** Python
* **Data Manipulation:** pandas, NumPy
* **Machine Learning Framework:** scikit-learn, XGBoost
* **Class Imbalance Resolution:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Hyperparameter Tuning:** GridSearchCV with 3-Fold Stratified Cross-Validation
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) KernelExplainer

---

## 📊 Pipeline Workflow

### 1. Data Integrity & Preprocessing
* Processed **12,330 session records** from online shopper behavioral data.
* Implemented strict stratified splitting to ensure consistent training and testing target distributions.
* **SMOTE Oversampling** was applied **strictly within the training folds** during cross-validation to eliminate data contamination, leakage, or evaluation biases.

### 2. Two-Layer Stacking Ensemble Architecture
* **Base Learners (Layer 0):** Random Forest (RF), Extra Trees (ET), and XGBoost.
* **Meta-Classifier (Layer 1):** Logistic Regression.
* **Optimization:** Executed a systematic hyperparameter sweep via `GridSearchCV` across all tree-based base learners to minimize variance.

### 3. Model Explainability (XAI)
* Deployed the **SHAP KernelExplainer** to decode global feature importance as well as local session-level predictions.
* Successfully mapped abstract computational boundaries into transparent indicators, evaluating metrics like `PageValues` and `ExitRates` to expose concrete consumer buying behavior.

---

## 📈 Performance Results
The proposed framework achieved state-of-the-art results, successfully outperforming standalone baseline models and previous literature benchmarks:

* **Classification Accuracy:** `90.23%`
* **ROC-AUC Score:** `0.9205`
* **Generalization Gap:** Restricted to a modest `3.24%`, indicating exceptionally stable real-world deployment metrics.

---

## 💡 Key Business Insights Captured
* **Primary Catalyst (`PageValues`):** High web page valuation immediately close to the checkout stream acts as the single strongest global predictor for positive buyer conversion.
* **Friction Metrics (`ExitRates`):** Micro-level user disengagement indicators (like exit and bounce rates) systematically drag down purchase probabilities, giving product teams concrete areas to optimize web flow.
