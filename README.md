# 🌲 Forest Cover Type Prediction

A complete multiclass machine learning project for predicting **forest cover types** from cartographic and environmental features.

The project follows an end-to-end machine learning workflow, including exploratory data analysis, preprocessing, outlier capping, Yeo-Johnson transformation, feature engineering, class-imbalance handling with SMOTE, feature selection, model comparison, XGBoost experimentation, final model evaluation, and a **Streamlit prediction application**.

---

## 📌 Project Overview

Forest ecosystems can be classified into different cover types based on environmental and cartographic characteristics.

The objective of this project is to build a machine learning model that predicts one of **7 forest cover types** from the available input features.

The project focuses not only on achieving good predictive performance, but also on building a complete and reproducible machine learning workflow from raw data to an interactive prediction application.

### Problem Type

* **Task:** Multiclass Classification
* **Number of Classes:** 7
* **Final Model:** XGBoost
* **Deployment:** Streamlit
* **Final Test Accuracy:** **95.12%**
* **Final Macro F1-Score:** **89.41%**
* **Final Weighted F1-Score:** **95.10%**

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Outlier Capping
     │
     ▼
Yeo-Johnson Transformation
     │
     ▼
Feature Engineering + Encoding
     │
     ▼
Normal SMOTE
(Training Data Only)
     │
     ▼
Feature Selection
     │
     ▼
Train 5 Models
     │
     ├── Logistic Regression
     ├── KNN
     ├── Decision Tree
     ├── Random Forest
     └── XGBoost
     │
     ▼
Validation / Test Comparison
     │
     ▼
Select XGBoost
     │
     ▼
XGBoost Hyperparameter Experiment
     │
     ▼
Compare Tuned vs Normal XGBoost
     │
     ▼
Normal XGBoost Selected
     │
     ▼
Final Evaluation
     │
     ▼
Streamlit Prediction Application
```

---

# 📊 Dataset

The dataset contains environmental and cartographic information used to classify forest cover types.

The target variable contains **7 different forest cover classes**.

The original dataset is highly imbalanced, with some cover types having significantly more observations than others.

Because of this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

> The validation and test datasets were kept untouched by SMOTE to provide an unbiased evaluation of the models.

---

# 🔍 Exploratory Data Analysis

The EDA phase was used to understand the dataset before model development.

The analysis included:

* Dataset structure
* Feature distributions
* Target-class distribution
* Missing-value analysis
* Duplicate analysis
* Numerical feature analysis
* Categorical/binary feature analysis
* Correlation analysis
* Outlier analysis
* Skewness analysis

The EDA results were used to determine the appropriate preprocessing and modeling strategy.

---

# 🧹 Data Preprocessing

## 1. Train / Validation / Test Split

The dataset was first divided into:

* Training set
* Validation set
* Test set

The split was performed **before preprocessing and feature engineering** to reduce the risk of information leakage.

---

## 2. Outlier Capping

Outliers were handled using **capping rather than removing observations**.

This allowed extreme values to be controlled while preserving the original observations in the dataset.

---

## 3. Yeo-Johnson Transformation

Skewed numerical features were transformed using the **Yeo-Johnson transformation**.

This was used to reduce skewness and make the numerical feature distributions more suitable for machine learning algorithms.

---

## 4. Feature Engineering and Encoding

Feature engineering was performed to prepare the dataset for machine learning.

This included:

* Creating useful derived features
* Transforming existing features
* Encoding required variables
* Preparing the final feature matrix

---

# ⚖️ Handling Class Imbalance

The original dataset contained significant class imbalance.

To address this, **Normal SMOTE** was applied to the **training data only**.

```text
Training Data
      │
      ▼
     SMOTE
      │
      ▼
Balanced Training Data
```

The validation and test datasets were **not oversampled**.

This ensures that final model performance is measured on data that represents the original distribution.

---

# 🎯 Feature Selection

After preprocessing and SMOTE, feature selection was performed to identify the final set of features used for model training.

The selected features were then used consistently for training and evaluation.

---

# 🤖 Models Compared

Five machine learning algorithms were trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. XGBoost

The models were compared using:

* Accuracy
* Macro F1-score
* Weighted F1-score

Macro F1 was particularly important because the dataset contains multiple classes with different sample sizes.

---

# 🏆 Final Model Comparison

The final models were evaluated on the untouched test set.

| Model               |   Accuracy |   Macro F1 | Weighted F1 |
| ------------------- | ---------: | ---------: | ----------: |
| **XGBoost**         | **95.12%** | **89.41%** |  **95.10%** |
| Random Forest       |     94.37% |     88.48% |      94.40% |
| Decision Tree       |     92.10% |     83.48% |      92.22% |
| KNN                 |     76.64% |     65.89% |      78.06% |
| Logistic Regression |     67.36% |     60.64% |      70.72% |

### Model Selection

**XGBoost** achieved the best overall performance across all three evaluation metrics.

Therefore, XGBoost was selected for further experimentation and final evaluation.

---

# ⚙️ XGBoost Hyperparameter Experiment

After selecting XGBoost as the best-performing algorithm, hyperparameter tuning was performed to investigate whether the model's performance could be improved.

The tuned XGBoost model was compared against the original XGBoost configuration.

The tuning experiment **did not outperform the original XGBoost model on the final evaluation**.

Therefore:

> **The original/normal XGBoost model was selected as the final model.**

This project does not claim that hyperparameter tuning automatically improves performance. The final model was selected based on the actual evaluation results.

---

# 🏅 Final XGBoost Performance

The final model achieved:

| Metric                |      Score |
| --------------------- | ---------: |
| **Test Accuracy**     | **95.12%** |
| **Macro F1-Score**    | **89.41%** |
| **Weighted F1-Score** | **95.10%** |

These results show strong overall performance across the seven-class classification problem.

The high macro F1-score also indicates that the model performs reasonably well across the different classes rather than relying only on the majority class.

---

# 📋 Final Classification Report

```text
              precision    recall  f1-score   support

           0       0.90      0.92      0.91       460
           1       0.90      0.96      0.93       324
           2       0.81      0.77      0.79       324
           3       0.92      0.98      0.95       324
           4       0.97      0.98      0.97     15461
           5       0.77      0.84      0.80       324
           6       0.93      0.89      0.91      4667

    accuracy                           0.95     21884
   macro avg       0.89      0.90      0.89     21884
weighted avg       0.95      0.95      0.95     21884
```

---

# 🔎 Final Model Observations

The final XGBoost model performed particularly well on:

* **Class 3:** 98% recall
* **Class 4:** 98% recall
* **Class 1:** 96% recall
* **Class 0:** 92% recall
* **Class 6:** 89% recall
* **Class 5:** 84% recall
* **Class 2:** 77% recall

The minority classes are naturally more challenging because they contain fewer test observations.

Classes **2 and 5** remain the most challenging classes for the final model, making them useful areas for future error analysis.

---

# 📈 Confusion Matrix

The final XGBoost confusion matrix:

```text
[[  423     0     4     0    26     7     0]
 [    0   310     6     0     0     8     0]
 [    3    13   248     0     1    59     0]
 [    1     0     0   316     1     0     6]
 [   28     0    14     2 15111     8   298]
 [    1    20    31     0     0   272     0]
 [   12     0     3    25   490     0  4137]]
```

The confusion matrix provides a detailed view of which forest cover classes are most frequently confused with one another.

---

# 🌐 Streamlit Application

The final XGBoost model was integrated into a **Streamlit application** for interactive predictions.

The application allows users to:

1. Enter the required input features.
2. Submit the input values.
3. Apply the saved preprocessing and feature-selection steps.
4. Generate a prediction using the final XGBoost model.
5. Display the predicted forest cover type.

### Prediction Flow

```text
User Input
    │
    ▼
Streamlit Interface
    │
    ▼
Saved Preprocessing
    │
    ▼
Feature Engineering / Selection
    │
    ▼
Final XGBoost Model
    │
    ▼
Forest Cover Type Prediction
```

The Streamlit application provides a simple interface for demonstrating the trained machine learning model outside the development notebook environment.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* imbalanced-learn

### Deployment

* Streamlit

### Development Environment

* Jupyter Notebook
* Python Virtual Environment

---

# 📚 Key Skills Demonstrated

Through this project, I gained practical experience in:

* Exploratory Data Analysis
* Data preprocessing
* Train/validation/test splitting
* Outlier handling
* Yeo-Johnson transformation
* Feature engineering
* Feature encoding
* Handling class imbalance
* SMOTE
* Feature selection
* Multiclass classification
* Model comparison
* Ensemble learning
* XGBoost
* Hyperparameter tuning
* Confusion matrix analysis
* Classification metrics
* Model selection
* Model serialization
* Streamlit deployment

---

# 💡 Key Learning

One of the important lessons from this project was that **the most complicated or heavily tuned model is not always the best final model**.

Although hyperparameter tuning was performed on XGBoost, the tuned model did not outperform the original XGBoost configuration in the final evaluation.

Therefore, the **original XGBoost model was selected based on actual performance rather than assuming that tuning would always improve the result**.

---

# 🔮 Future Improvements

Possible future extensions include:

* SHAP-based model explainability
* More detailed error analysis for difficult classes
* REST API deployment
* Experiment tracking
* Automated model retraining
* Cloud deployment
* Improved Streamlit visualization

These are optional extensions and are not required for the current completed project.

---

# 👨‍💻 Author

**Saravanan**

Data Science & Python Enthusiast

---

## ⭐ Project Summary

This project demonstrates a complete machine learning workflow for a **7-class forest cover classification problem**, from exploratory data analysis and preprocessing to model comparison, final model selection, and interactive Streamlit deployment.

The final **XGBoost model achieved 95.12% test accuracy, 89.41% macro F1-score, and 95.10% weighted F1-score** on the untouched test set.
