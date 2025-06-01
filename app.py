import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
import streamlit.components.v1 as components
import altair as alt
import shap
import names
import pickle
import json
import numpy as np
import os
import redis

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = redis.StrictRedis.from_url(redis_url)



# API_URL = "http://127.0.0.1:8000/"

API_URL = "https://api-myapp.herokuapp.com/"

TIMEOUT = (5, 30)
CLASSES_NAMES = ['REPAY SUCCESS', 'REPAY FAILURE']
CLASSES_COLORS = ['green', 'red']
  
    
#Return altair chart of logistic regression feature importances    
@st.cache_data()
def LogisticRegression_importances_chart(model, feature_names):
    imp_df = pd.DataFrame({
        'features': feature_names,
        'importances': np.abs(model.coef_[0])
    })
    
    imp_sorted = imp_df.sort_values(by='importances', ascending=False)
    
    imp_chart = alt.Chart(imp_sorted.head(15), title="Top 15 feature importances").mark_bar().encode(
        x='importances',
        y=alt.Y('features', sort=None, title='features')
    )
    return imp_chart

#Get list of customers ID
@st.cache_data()
def get_cust_ids():
    response = requests.get(API_URL + "ids/", timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['ids']

@st.cache_data()
def get_cust_columns(cust_id):
    response = requests.get(API_URL + "columns/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return pd.Series(content)

#Get customers main columns mean values
@st.cache_data()
def get_columns_mean():
    response = requests.get(API_URL + "columns/mean", timeout=TIMEOUT)
    content = json.loads(response.content)
    return pd.Series(content)

#Get customers neighbors main columns mean values
@st.cache_data()
def get_columns_neighbors(cust_id):
    response = requests.get(API_URL + "columns/neighbors/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return pd.Series(content)

#Get customer prediction (class 1 : repay failure)
@st.cache_data()
def get_predictions(cust_id):
    response = requests.get(API_URL + "predict/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return content

#Get all customers SHAP values
@st.cache_data()
def get_shap_values():
    response = requests.get(API_URL + "shap", timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']),
                                   np.asarray(content['base_values']),
                                   feature_names=content['features'])
    return explanation

#Get customer SHAP explanation
@st.cache_data()
def get_shap_explanation(SK_ID_CURR):
    response = requests.get(API_URL + "shap/SK_ID_CURR=" + str(SK_ID_CURR), timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']), 
                                   content['base_values'],
                                   feature_names=content['features'])
    return explanation  


#Get feature importance
def get_feature_importances():
    response = requests.get(API_URL + "importances", timeout=TIMEOUT)
    content = json.loads(response.content)
    return pd.DataFrame(content)

#Create array of random names
@st.cache_data()
def create_customer_names(cust_numbers):
    return [names.get_full_name() for _ in range(cust_numbers)]
    
#Create a shap html component
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

st.markdown("""
<style>
body {
    background-color: #59ba6b;
}
</style>
""", unsafe_allow_html=True)

...
st.markdown("""
<style>
body {
    background-color: #59ba6b;
}
h1 {
    color: blue;
}
button {
    color: white;
    background-color: tomato;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

...

st.title('Interactive Dashboard')
st.header('Predict Loan Default Risk')
st.image("https://webnews.bg/uploads/images/50/1650/291650/768x432.jpg?_=1493804010")
    
# # Sidebar settings
st.sidebar.subheader("Settings")

# # Select the prediction threshold
pred_thresh = st.sidebar.slider("Prediction threshold : ", 0.15, 0.50, value=0.50, step=0.05,
                                help="Threshold of the prediction for class 1 : repay failure (standard=0.5)")
# Select type of explanation
shap_plot_type = st.sidebar.radio("Select the SHAP plot type :", ('Waterfall', 'Bar'),
                                  help="Type of plot for the SHAP explanation")
# Select source of feature importance
feat_imp_source = st.sidebar.radio("Feature importances source:", ('Model', 'SHAP'))

#  Create tabs
tab_single, tab_all = st.tabs(["Customer", "All customers"])

st.header('Prêt à Spend')


# General tab
with tab_all:

    st.subheader("Feature importances (" + feat_imp_source + ")")
    st.write("")
    
    if feat_imp_source == 'Logistic Regression':
        # Use the API to fetch feature importances for logistic regression
        st.altair_chart(LogisticRegression_importances_chart(), use_container_width=True)
        response = requests.get(API_URL + "importances")
        importances_df = pd.read_json(response.content)
        
        # Convert to altair chart and display
        chart = alt.Chart(importances_df.reset_index()).mark_bar().encode(
            x='shap_importance',
            y=alt.Y('index', sort='-x', title='Feature')
        ).properties(title="Top 15 feature importances")
        st.altair_chart(chart, use_container_width=True)

        expander = st.expander("About the feature importances..")
        expander.write("The feature importances displayed is computed from the trained Logistic Regression model.")
    else:
        # Display SHAP feature importance as before
        shap_values = get_shap_values()
        fig, _ = plt.subplots()
        fig.suptitle('Top 15 feature importances (test set)', fontsize=18)
        shap.summary_plot(shap_values, max_display=15, plot_type='bar', plot_size=[12, 6], show=False)
       
        st.pyplot(fig)
        expander = st.expander("About the feature importances..")
        expander.write("The feature importances displayed is computed from the SHAP values of the new customers. (test data)")

    
# Specific customer tab
with tab_single:
    # Get customer ids
    cust_ids = get_cust_ids()
    cust_names = create_customer_names(len(cust_ids))

    # Select the customer
    cust_select_id = st.selectbox(
        "Select the customer",
        cust_ids,
        format_func=lambda x: str(x) + " - " + cust_names[x])

    # Display the columns
    st.subheader("Customer information")
    cust_df = get_cust_columns(cust_select_id).rename(cust_names[cust_select_id])
    neighbors_df = get_columns_neighbors(cust_select_id).rename("Neighbors average")
    mean_df = get_columns_mean().rename("Customer")
    st.dataframe(pd.concat([neighbors_df, mean_df], axis=1))
   
    
    
    # Display prediction
    st.subheader("Customer prediction")
    response = requests.get(API_URL + f"predict/id={cust_select_id}")
    predictions = response.json()
    
    pred = (predictions['proba'] >= pred_thresh)
    pred_text = "**:" + CLASSES_COLORS[pred] + "[" + CLASSES_NAMES[pred] + "]**"
    st.markdown("The model prediction is " + pred_text)
    probability = 1 - round(predictions['proba'], 2)  # probability of repay (class 0)
    delta = probability - 1 + pred_thresh
    st.metric(label="Probability to repay", value=probability, delta=round(delta, 2))

    # Display SHAP force plot
    response = requests.get(API_URL + f"shap/id={cust_select_id}")
    shap_data = response.json()
    shap_explanation = shap.Explanation(
        np.asarray(shap_data['values']),
        base_values=shap_data['base_values'],
        feature_names=shap_data['features']
    )
    st_shap(shap.force_plot(shap_explanation))

    # Display shap bar/waterfall plot
    fig, _ = plt.subplots()
    if shap_plot_type == 'Waterfall':
        shap.plots.waterfall(shap_explanation, show=False)
    else:
        shap.plots.bar(shap_explanation, show=False)
    plt.title("Shap explanation plot", fontsize=16)
    fig.set_figheight(6)
    fig.set_figwidth(9)
    st.pyplot(fig)

    # Display some information
    expander = st.expander("About the SHAP explanation...")
    expander.write("The above plot displays the explanations for the individual prediction of the customer. \
                    It shows the postive and negative contribution of the features. \
                    The final SHAP value is not equal to the prediction probability.")
    
st.image("https://www.nesto.ca/wp-content/uploads/2022/06/template-020.png")
st.write("")




