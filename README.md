# Mutual Fund Risk Classification using Machine Learning

---

## a. Problem Statement

Investors often struggle to understand the risk level of mutual funds based on multiple financial indicators such as returns, ratings, NAV, AUM, and fund category. 

The objective of this project is to build Machine Learning classification models to automatically classify mutual funds into risk categories:

- Low
- Moderate
- High
- Very High

The project also includes deployment of these models using Streamlit to provide an interactive prediction and evaluation interface.

---

## b. Dataset Description

The dataset used is a Mutual Fund NAV and Returns dataset containing more than 500 records and more than 12 features.

### Dataset Features

| Feature | Description |
|---|---|
| AMC | Asset Management Company |
| Morning Star Rating | Fund performance rating |
| Value Research Rating | Independent rating |
| 1 Month Return | Short term return |
| NAV | Net Asset Value |
| 1 Year Return | Annual performance |
| 3 Year Return | Long term performance |
| Minimum Investment | Minimum amount required |
| AUM | Assets Under Management |
| Category | Fund category |
| Risk | Target variable |

### Data Preprocessing Steps

- Removed invalid AUM values
- Converted percentage columns to numeric
- Encoded categorical variables (AMC, Category)
- Standardized numerical features using StandardScaler
- Cleaned and encoded Risk target column

---

## c. Models Used and Evaluation Metrics

The following 6 Machine Learning models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.64 | 0.84 | 0.53 | 0.52 | 0.52 | 0.46 |
| Decision Tree | 0.76 | 0.81 | 0.76 | 0.71 | 0.73 | 0.65 |
| KNN | 0.74 | 0.90 | 0.72 | 0.71 | 0.71 | 0.62 |
| Naive Bayes | 0.46 | 0.79 | 0.63 | 0.44 | 0.31 | 0.31 |
| Random Forest (Ensemble) | 0.85 | 0.96 | 0.86 | 0.84 | 0.85 | 0.78 |
| XGBoost (Ensemble) | 0.84 | 0.95 | 0.84 | 0.82 | 0.83 | 0.77 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provided stable baseline performance but struggled with non-linear relationships. |
| Decision Tree | Captured non-linear patterns but prone to overfitting. |
| KNN | Performed well but sensitive to feature scaling and dataset size. |
| Naive Bayes | Lowest performance due to strong independence assumptions. |
| Random Forest (Ensemble) | Best overall performance due to ensemble learning and variance reduction. |
| XGBoost (Ensemble) | High performance and strong generalization but required environment-specific handling during deployment. |

---

## Streamlit Application Features

The deployed Streamlit app includes:

- CSV Dataset Upload (Test Data Only)
- Model Selection Dropdown
- Evaluation Metrics Display
- Confusion Matrix Visualization
- Classification Report Display
- Prediction Generation
- Prediction CSV Download

---

## Deployment Links

### Streamlit App
https://mutual-fund-risk-classifier-hiwzetzuy688lb9zcmxdfg.streamlit.app/

### GitHub Repository
https://github.com/Gagandeep9603/mutual-fund-risk-classifier/tree/main/data

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

---

## Evaluation Metrics Used

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Author

Gagandeep Bhatia