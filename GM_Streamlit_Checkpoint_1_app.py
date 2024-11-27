import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle as pkl
import streamlit as st
import numpy as np
import pandas as pd

# load model from pickle file
loaded_pipeline = joblib.load('best_model_pipeline.pkl')

# Load categorical features to use as options
with open("regions.pkl", "rb") as file:
    regions= pkl.load(file)

with open("tenures.pkl", "rb") as file:
    tenures= pkl.load(file)

# Title
st.title("Express Churn Predictor")
st.markdown("---")

# Inputs
st.text("Enter your inputs below to predict your churn")

# Regions
region= st.selectbox("Select Region", options=regions, placeholder="Select Region", key="regions")

# Tenure
tenure= st.selectbox("TENURE", options=tenures, placeholder="Select Tenure", key="tenures")

montant= st.slider("MONTANT", 0.0, 500000.0, 100.0)

freq_rech= st.slider("FREQUENCE_RECH", 1.0, 200.0, 10.0 )

revenue= st.slider("REVENUE", 1.0 , 1000000.0, 1000.0)

segement= st.slider("APPU_SEGMENT", 0.0, 200000.0, 100.0)

freq= st.slider("FREQUENCE", 1.0, 100.0, 1.0)

data_volume= st.slider("DATA_VOLUME", 0.0, 600000.0, 100.0)

on_net= st.slider("ON_NET", 0.0, 200000.0, 100.0)

orange= st.slider("ORANGE", 0.0, 10000.0, 10.0)

tigo= st.slider("TIGO", 0.0, 5000.0, 10.0)

zone1= st.slider("ZONE1", 0.0, 3000.0 , 10.0)

zone2= st.slider("ZONE2", 0.0, 3000.0, 10.0)

reg= st.slider("REGULARITY", 1.0, 100.0, 1.0)

freq_top_track= st.slider("FREQ_TOP_PACK", 1.0, 600.0, 1.0)

# Format input data
input_data= np.array([region, tenure, montant, freq_rech, revenue, segement, freq, data_volume, on_net, orange, tigo, zone1, zone2, reg, freq_top_track]).reshape(1,-1)
input_df= pd.DataFrame(input_data, columns=['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
       'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK'])


# Predict churn
btn= st.button("Predict Churn")
if btn:
    churn= loaded_pipeline.predict(input_df)
    if churn:
        st.success("CHURNED")
    elif not churn:
        st.info("DID NOT CHURN")
    else:
        st.error("Something went wrong")






