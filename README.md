## HomeCredit Insight: Scoring & Predictive Analytics Tool

# Credit Scoring System: "Prêt à dépenser"

## Overview

This project is tailored for "Prêt à dépenser", a financial institution offering consumer loans primarily to those with little to no credit history. Our main objective is to design and implement a scoring tool to predict the likelihood of a customer repaying their loan and to present this information through an interactive dashboard.

### Home Credit Default Risk

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. By understanding the potential default risks of the clients, the company can ensure both the safety of the loan and the client's ability to repay it.

The challenge is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. Such insights can be used to empower loan approvers to make informed decisions and help more people get the right level of sustainable credit.



## Data

The provided datasets contain information about the clients' previous credit history, socio-economic indicators, and other relevant data. The main data files include:

* `application_{train|test}.csv`: Main training and testing data with information about each loan application at Home Credit.
* `bureau.csv`: Data concerning the client's previous credits from other financial institutions.
* `bureau_balance.csv`: Monthly balance data for the client's previous credits from other financial institutions.
* `previous_application.csv`: Previous applications for loans at Home Credit.
* `POS_CASH_balance.csv`: Monthly balance snapshots of previous point of sale or cash loans.
* `credit_card_balance.csv`: Monthly balance snapshots of the client's credit cards.
* `installments_payments.csv`: Repayment history for previous Home Credit loans.

Each application has an associated label indicating the client's repayment abilities.



## Development Process

* **Data Analysis**: Exploration of data tables, attribute understanding, and correlation discovery.
* **Data Processing**: Custom functions for preprocessing and feature engineering.
* **Model Experiments**: LogisticRegression, RandomForestClassifier, BalancedRandomForestClassifier — LogisticRegression (balanced) performed best.
* **Model Optimization**: Threshold tuning for improved prediction.
* **Data Drift Monitoring**: With `evidently` to ensure model relevance.
* **Experiment Tracking**: Tracked via `MLFlow UI`.
* **Dashboard**: Built using Streamlit.
* **Deployment**: API served via FastAPI, hosted on PythonAnywhere.



## Tools Used

* **Modeling**: `LogisticRegression`, `RandomForestClassifier`, `BalancedRandomForestClassifier`
* **Optimization**: `GridSearchCV`
* **Monitoring**: `evidently`
* **Tracking**: `MLFlow UI`
* **Dashboard**: `Streamlit`
* **Deployment**: `FastAPI`, `PythonAnywhere`



## Application Link

🌐 [https://api-myapp.herokuapp.com/](https://api-myapp.herokuapp.com/)



## Next Steps

* Improve preprocessing speed
* Integrate new data sources
* Conduct A/B testing on model impact


