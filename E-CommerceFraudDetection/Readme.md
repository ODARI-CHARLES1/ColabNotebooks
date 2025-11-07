# E-Commerce Fraud Detection

## Overview

In 2024, e-commerce platforms across Istanbul, Berlin, New York, London, and Paris began noticing strange transaction bursts. Some cards tested with $1 purchases at midnight. Others shipped "gaming accessories" 5,000 km away. Promo codes were being reused from freshly created accounts.

To investigate these global patterns safely, this synthetic dataset recreates realistic fraud behavior across countries, channels, and user profiles — allowing anyone to build, test, and compare fraud-detection models without exposing any real user data.

This project provides a comprehensive synthetic dataset for machine learning practitioners to develop and evaluate fraud detection algorithms. It simulates real-world e-commerce transaction patterns, including legitimate and fraudulent activities, to facilitate research and model development in anomaly detection and classification.

## Objectives

- Provide a realistic, synthetic dataset for fraud detection research
- Enable the development and testing of machine learning models for identifying fraudulent transactions
- Facilitate behavioral analysis through multi-transaction user profiles
- Support cross-validation of fraud detection techniques across different geographical and temporal contexts
- Promote safe experimentation without compromising real user data privacy

## Dataset Description

The dataset contains approximately 300,000 transactions from 6,000 unique users, with each user performing between 40-60 transactions. This volume allows for robust statistical analysis and machine learning model training.

### Key Characteristics

- **Scale**: 6,000 unique users, ~300,000 transactions
- **User Behavior**: Multiple transactions per user (40–60) enabling behavioral pattern analysis
- **Feature Engineering**: Strong correlations between features, mimicking real-world data relationships
- **Geographical Diversity**: Cross-country dynamics with variables for transaction country and card issuing country (bin_country)
- **Class Imbalance**: Natural fraud rate of ~2%, reflecting real financial system distributions
- **Temporal Patterns**: Realistic time-based behaviors including night-time fraud spikes and daily transaction rhythms
- **Explainability**: Features designed for easy visualization, modeling, and interpretation

### Data Structure

The dataset includes features such as:
- User identifiers and transaction timestamps
- Transaction amounts and merchant information
- Geographical data (countries, distances)
- Payment method details
- Behavioral patterns (frequency, amounts, locations)
- Fraud labels for supervised learning

## Features

- **Synthetic Data Generation**: Algorithmically created to match real fraud patterns without using actual user data
- **Behavioral Analysis**: Supports pattern recognition across user transaction histories
- **Geospatial Analysis**: Enables modeling of location-based fraud indicators
- **Temporal Modeling**: Incorporates time-series elements for detecting unusual timing patterns
- **Imbalanced Classification**: Realistic fraud ratios for testing anomaly detection algorithms
- **Feature Interpretability**: Clear, correlated features for model explainability and debugging

## How to Use

1. **Download the Dataset**: Obtain the synthetic transaction data from the project repository
2. **Data Exploration**: Use Python libraries like Pandas and Matplotlib for initial analysis
3. **Preprocessing**: Clean and prepare data for machine learning (handling missing values, encoding categorical variables)
4. **Model Development**: Apply classification algorithms such as:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks
   - Anomaly Detection methods (Isolation Forest, Autoencoders)
5. **Evaluation**: Use metrics appropriate for imbalanced datasets:
   - Precision, Recall, F1-Score
   - AUC-ROC
   - Precision-Recall curves
   - Confusion Matrix analysis

### Sample Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('ecommerce_fraud_data.csv')

# Preprocessing
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

## Applications

- Fraud detection model development and benchmarking
- Behavioral analytics research
- Machine learning education and experimentation
- Algorithm comparison across different fraud detection techniques
- Privacy-preserving data science demonstrations

## References

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Imbalanced Classification Techniques](https://machinelearningmastery.com/imbalanced-classification/)
- [Fraud Detection Research Papers](https://scholar.google.com/)

## License

This project is released under the MIT License. The synthetic dataset is provided for educational and research purposes.

## Follow Me

[![GitHub followers](https://img.shields.io/github/followers/CharlesOdari?label=Follow&style=social)](https://github.com/ODARI-CHARLES1)
[![Portfolio](https://img.shields.io/badge/Portfolio-View-blue?logo=google-chrome)](https://charles.k.odari.portfolio.thegtm.or.ke/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://ke.linkedin.com/in/odari-kibisi-charles-329b19331)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:daymondodari68@gmail.com)
