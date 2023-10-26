from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, root_validator
from typing import Optional
import pickle
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import ssl
import dill
import uvicorn
from fastapi.responses import HTMLResponse
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

with open('/Users/innakonar/Desktop/Project_7/best_model_filename.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
# Deserialize SHAP explainer    
with open('/Users/innakonar/Desktop/Project_7/explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)
app = FastAPI()

N_CUSTOMERS = 1000
N_NEIGHBORS = 15

MAIN_COLUMNS = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
                'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
                'AMT_INCOME_TOTAL','DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE']
CUSTOM_THRESHOLD = 0.7

# Get test dataframe
test_df = pd.read_csv("/Users/innakonar/Desktop/Project_7/test_feature_engineering.csv")
test_columns = test_df.columns

# Before calling prepare_data:
def preprocess_data(df, template_df=test_df):
    # One-hot encoding with all possible categories from template_df
    full_encoded = pd.get_dummies(template_df)
    df_encoded = pd.get_dummies(df).reindex(columns=full_encoded.columns, fill_value=0)

    # Imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(df_imputed, columns=df_encoded.columns)
    
    # Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    
    return df_scaled

prep_df = test_df.iloc[0: N_CUSTOMERS].copy()
data_encoded = preprocess_data(prep_df)

def prepare_data(data, data_encoded, n_neigbhors, n_customers):
    """Prepare the data, find the nearest neighbors and compute the shap values."""
    
    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neigbhors, algorithm='ball_tree').fit(data_encoded)
    _, neighbors_indices = neighbors.kneighbors(data_encoded)
    
    # Compute shap values
    shap_values = explainer(data_encoded)
    
    return prep_df, neighbors_indices, shap_values

prep_df, neighbors_indices, shap_values = prepare_data(test_df, data_encoded, N_NEIGHBORS, N_CUSTOMERS)


@app.get('/')
def main():
    """ API main page """
    return "Hello There! This is the front page of the scoring API."


@app.get("/ids")
def ids():
    """ Return the customers ids """
    return {'ids': test_df.head(N_CUSTOMERS).index.to_list()}


@app.get("/columns/id={cust_id}")
def columns(cust_id: int):
    """ Return the customer main columns values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    cust_main_df = prep_df.iloc[cust_id][MAIN_COLUMNS]
    return cust_main_df.to_json()


@app.get("/columns/mean")
def colmuns_mean():
    """Return the entire dataset main columns mode values"""
    return prep_df[MAIN_COLUMNS].mode().iloc[0].to_dict()


@app.get("/columns/neighbors/id={cust_id}")
def colmuns_neighbors(cust_id: int):
    """Return the 15 nearest neighbors main columns mode values"""
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    
    neighbors_data = prep_df.iloc[neighbors_indices[cust_id]]
    
    # Calculate mode for all columns
    modes = neighbors_data.mode().iloc[0].to_dict()
    
    return modes


@app.get("/predict/id={cust_id}")
def predict(cust_id: int):
    """ Return the customer predictions of repay failure (class 1) """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    row = pd.DataFrame(prep_df.iloc[cust_id]).T  
    processed_row = preprocess_data(row)
    proba = model.predict_proba(processed_row)[0][1]  # prediction of class 1
    return {'proba': proba.tolist()}



@app.get("/shap")
def explain_all():
    """ Return all shap values """
    return {'values': shap_values.values.tolist(),
            'base_values': shap_values.base_values.tolist(),
            'features': explainer.feature_names}


@app.get("/shap/id={cust_id}")
def explain(cust_id: int):
    """ Return the customer shap values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    return {'values': shap_values[cust_id].values.tolist(),
            'base_values': float(shap_values[cust_id].base_values),
            'features': explainer.feature_names}

@app.get("/importances")
def importances():
    """ Return the top 15 feature importances based on SHAP values """
    shap_sum = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame(data=shap_sum, index=test_columns, columns=['shap_importance'])
    imp_df = imp_df.sort_values(by='shap_importance', ascending=False).head(15)
    return imp_df.to_json()


if __name__ == "__main__":
    uvicorn.run("scoring_api:app", reload=True, host="0.0.0.0", port=8000)
