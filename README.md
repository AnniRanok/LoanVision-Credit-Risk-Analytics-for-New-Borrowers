# OC_Projet_7
## HomeCredit Insight: Scoring &amp; Predictive Analytics Tool

# Credit Scoring System: "Prêt à dépenser"   
## Overview  
This project is tailored for "Prêt à dépenser", a financial institution offering consumer loans primarily to those with little to no credit history. Our main objective is to design and implement a scoring tool to predict the likelihood of a customer repaying their loan and to present this information through an interactive dashboard.

### Home Credit Default Risk:  
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. By understanding the potential default risks of the clients, the company can ensure both the safety of the loan and the client's ability to repay it.

The challenge is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. Such insights can be used to empower loan approvers to make informed decisions and help more people get the right level of sustainable credit.

## Data  
The provided datasets contain information about the clients' previous credit history, socio-economic indicators, and other relevant data. The main data files include:

- **application_{train|test}.csv**: Main training and testing data with information about each loan application at Home Credit.
- **bureau.csv**: Data concerning the client's previous credits from other financial institutions. Each row represents one credit from the client's credit history, merged with the main dataset on the SK_ID_CURR key.
- **bureau_balance.csv**: Monthly balance data for the client's previous credits from other financial institutions.
- **previous_application.csv**: Previous applications for loans at Home Credit for clients who have loans in the main dataset (application_train|test).
- **POS_CASH_balance.csv**: Monthly balance snapshots of previous point of sale or cash loans that the client had with Home Credit.
- **credit_card_balance.csv**: Monthly balance snapshots of the client's credit cards with Home Credit.
- **installments_payments.csv**: Repayment history for the previously disbursed credits in Home Credit related to the loans in the main dataset.  
Each application has an associated label indicating the client's repayment abilities.

## Development Process  
- **Data Analysis**: Exploration of the available data tables, understanding their attributes, and identifying correlations.
- **Data Processing**: Creation of functions for data preprocessing, feature generation, and more.
- **Model Experiment**s: Comparative analysis between **LogisticRegression**, **RandomForestClassifier**, and **BalancedRandomForestClassifier models**. Chose the best-performing model - **LogisticRegression** with class_weight='balanced'.
- **Model Optimization**: Set an optimal threshold value to maximize the model's effectiveness.
- **Data Drift Monitorin**g: Regular checks for potential changes in data distribution using evidently.
- **Visualization Tracking**: Visual tracking of each experiment's details through **MLFlow UI**.
- **Dashboard Developmen**t: Crafted an interactive dashboard using **Streamlit**.
- **Deployment**: Deployed the model and dashboard on **PythonAnywhere** through **FastAPI**.

## Tools Used
- **Modeling**: LogisticRegression, RandomForestClassifier, BalancedRandomForestClassifier
- **Optimization**: GridSearchCV
- **Data Drift Monitoring**: evidently
- **Logging**: MLFlow UI
- **Dashboard**: Streamlit
- **API & Deployment**: FastAPI, PythonAnywhere

## Links
- **Application** :https://api-myapp.herokuapp.com/

## Next Steps
Improve data processing for quicker predictions.
Integrate new data sources to enhance the model's accuracy.
Conduct A/B testing to measure the impact of the model on business metrics.
