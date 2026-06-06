# Credit Risk Modeling & Scoring System (Home Credit Insight)

## Overview

This project implements a credit risk scoring and predictive modeling pipeline for assessing loan repayment probability based on historical application data.

The system explores supervised machine learning approaches for credit default prediction using structured financial and behavioral features.

This is a machine learning engineering project focused on model development, evaluation, and monitoring rather than production deployment.


## Problem Statement

The objective is to estimate the probability of default for loan applicants with limited or no credit history, using multi-source financial and behavioral data.

The output supports risk-based decision-making in consumer lending scenarios.


## Data Sources

The dataset consists of multiple relational tables commonly used in credit risk modeling:

- Loan application data (train/test sets)
- Bureau credit history data
- Previous loan applications
- Monthly repayment and balance histories
- Credit card usage data
- Installment repayment records

These datasets are used for feature engineering and predictive modeling.


## Methodology

The project follows a standard credit risk modeling pipeline:

- Data exploration and schema analysis  
- Feature engineering from multi-table relational data  
- Data preprocessing and cleaning  
- Model training and evaluation  
- Threshold optimization for classification output  
- Model monitoring and drift analysis  


## Models Evaluated

- Logistic Regression (balanced class weights)  
- Random Forest Classifier  
- Balanced Random Forest Classifier  

Final model selection was based on classification performance and stability.


## Experimentation & Tracking

- Hyperparameter tuning using GridSearchCV  
- Experiment tracking with MLflow  
- Data drift monitoring using Evidently AI  


## Evaluation Approach

Model performance was evaluated using classification metrics with threshold tuning to optimize trade-off between precision and recall in credit risk context.


## Interface Layer

- Streamlit dashboard for interactive analysis and model interpretation  
- API layer using FastAPI (experimental deployment setup)  


## Limitations

- Not deployed in a production banking environment  
- Requires further calibration for real-world credit portfolios  
- Limited external validation on unseen financial institutions  


## Tech Stack

- Python (Pandas, NumPy, Scikit-learn)  
- Machine Learning: Logistic Regression, Random Forest, Balanced RF  
- Model Optimization: GridSearchCV  
- Experiment Tracking: MLflow  
- Monitoring: Evidently AI  
- Dashboard: Streamlit  
- API Layer: FastAPI  


## Status

This project developed as a machine learning prototype for credit risk modeling and model development workflows.


## Key Focus Areas

- Credit scoring model development  
- Multi-source financial feature engineering  
- Model evaluation under class imbalance  
- Monitoring and experiment tracking workflows  


