import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
from zipfile import ZipFile
from sklearn.cluster import KMeans

import base64
import streamlit.components.v1 as components

from joblib import load




#Load the saved model and data
model = load("LightGBMC.joblib")
df= pd.read_csv("app_streamlit_df.csv", sep=',')
threshold = 0.1685
id_client = df["SK_ID_CURR"].tolist()

columns_description = pd.read_csv('HomeCredit_columns_description.csv', sep=',', encoding = 'iso8859_1')
list_feature = columns_description.Row 

feats = [f for f in df.columns if f not in ['TARGET','TARGET_proba', "kmeans_label",'SK_ID_CURR','index','SK_ID_BUREAU','SK_ID_PREV']]
X = df[feats]
y = df["TARGET"]



st.set_page_config( 
    page_title="Loan Prediction App",
    page_icon="logo.png" 
)
st.set_option('deprecation.showPyplotGlobalUse', False)



######################
#main page layout
######################

st.title("Crédit de consommation")
st.subheader("L'application de  machine learning va vous aider à faire la décision!")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("logo.png")

with col2:
    st.write("""Cette application permet de prédire l'accord d'un crédit pour les clients ayant très peu d'informations
             à la l'entreprise "Prêt à dépenser".""")



st.subheader("Voici les résultats de prédiction: ")

######################
#Fonctions
######################
@st.cache 
def idx_client(sk_id):
    return np.where(df.SK_ID_CURR==sk_id)[0][0]


@st.cache 
def pie_client(sk_id):
    labels = ['Accord Crédit','Refus Crédit']
    a = df.iloc[idx_client(sk_id)]["TARGET_proba"]
    b = 1-a 
    # pull is given as a fraction of the pie radius
    fig = go.Figure(data=[go.Pie(labels=labels, values=[ float(b), float(a)], pull=[0, 0.2])])
    return fig

@st.cache 
def client_target(sk_id):
   return df.iloc[idx_client(sk_id)]["TARGET"]
   

@st.cache 
def client_cluster(sk_id):
   num= df.iloc[idx_client(sk_id)]["kmeans_label"]
   data = df.loc[df["kmeans_label"]==num].sample(10)
   return data[list_fi()]
   

 
@st.cache(suppress_st_warning=True)
def shap_valu(sk_id):
    st.subheader('Interpretabilité des Résultats')
    shap.initjs()
    idx = idx_client(sk_id)
    test_1 = X.iloc[idx]
    # explain the model's predictions using SHAP
    # Create a tree explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    #fig = shap.summary_plot(shap_values[1], X.values, feature_names = feats)
    fig= shap.waterfall_plot(shap.Explanation(values=shap_values[1][idx], 
                                       base_values=explainer.expected_value[1], data=test_1,  
                                     feature_names = feats))
    st.pyplot(fig)

@st.cache
def feature_descriptions(choix_feature):
    descprition = columns_description.loc[columns_description.Row == choix_feature]["Description"]
    return descprition

@st.cache
def list_fi():
    fi_df =pd.read_csv("feature_importance_df.csv", sep=',')
    cols = fi_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_fi = fi_df.loc[fi_df.feature.isin(cols)]
    list_features = best_fi.sort_values(by="importance" , ascending=False)[:100]["feature"].unique().tolist()
    return list_features

######################
#sidebar layout
######################
st.sidebar.title("Application ")

sk_id =st.sidebar.selectbox('Please choose SK_ID_CURR', id_client)


client_info = st.sidebar.radio('Client info', ("Oui", "Non"))
client_similaires = st.sidebar.radio('Clients similaires', ("Oui", "Non"))





######################
#main page layout
######################
st.write("SK_ID_CURR est :", sk_id)

st.write("Le client est avec label :", client_target(sk_id))
st.write("Seuil : ", threshold)
st.plotly_chart( pie_client(sk_id), use_container_width=True)
        
if client_info == 'Oui':
    st.image(shap_valu(sk_id))
else:
    st.write("Vous pouvez voir plus de détails.")
    


if client_similaires == 'Oui':
    st.write(client_cluster(sk_id))
else:
    st.write("Vous pouvez afficher les clients simillaires.")



container = st.container()
choix_feature =st.selectbox('Choisire un feature', list_feature)
container.write( feature_descriptions(choix_feature)




    
