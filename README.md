# Airbnb Superhost Prediction

## Overview
This machine learning project predicts whether an Airbnb host is a Superhost using Logistic Regression.

The project covers the complete machine learning workflow, including data preparation, model training, hyperparameter tuning, model evaluation, feature selection, and model persistence.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Machine Learning Workflow
1. Loaded and prepared the Airbnb dataset
2. Split the data into training and testing sets
3. Trained a baseline Logistic Regression model
4. Used GridSearchCV with 5-fold cross-validation to tune the C hyperparameter
5. Evaluated models using Precision-Recall curves, ROC curves, and AUC
6. Used SelectKBest to identify the top 5 features
7. Saved the trained model using Pickle

## Results
- Baseline AUC: 0.8228
- Optimized AUC: 0.8236
- Top 5 Features AUC: approximately 0.7972

The optimized Logistic Regression model slightly improved the AUC compared with the baseline model.

## Top 5 Selected Features
- host_response_rate
- number_of_reviews
- number_of_reviews_ltm
- number_of_reviews_l30d
- review_scores_cleanliness

## Files
- `Airbnb_Superhost_Prediction.ipynb` - Complete machine learning analysis
- `airbnbData_train.csv` - Dataset used for model development
- `model_best.pkl` - Saved trained Logistic Regression model

## What I Learned
Through this project, I practiced building and evaluating a classification model, tuning hyperparameters with cross-validation, selecting important features, comparing model performance using AUC and ROC curves, and saving a trained machine learning model for future use.