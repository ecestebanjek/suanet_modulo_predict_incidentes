import streamlit as st
import pandas as pd
import geopandas as gpd
import time
from shapely.ops import cascaded_union
from shapely.geometry import Point
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import random
import folium
import time
from IPython.display import display, clear_output
import plotly.express as px
#from streamlit_folium import st_folium, folium_static
from PIL import Image

import datetime as dt
import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GRU, Concatenate,  Dropout, Masking
import keras
from keras import layers
import os
import math
import shapefile as shp
from sklearn.preprocessing import MinMaxScaler
import folium
import os

### Libreria de simulaciones
import lib.pred_library as pred

################################ SET PAGE
st.set_page_config(
    page_title="SUANET - Modelo predictivo de incidentes",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "Este es el modulo 3 del aplicativo SUANET para predicción de incidentes"
    }
)


################################# PAGINA DE INICIO
## Show in webpage
st.markdown("# MODELO PREDICTIVO DE INCIDENTES")
with st.expander("Ver explicación"):
    st.markdown(
        """
        # MODELO PREDICTIVO DE INCIDENTES
        
        ## Explicación
        
        Se plantea un modelo que mezcla dos tipos de redes neuronales:
        - CNN: Red neuronal convolucional, mayormente usada para capturar los efectos espaciales de los datos y usadas principalmente para el análisis de imágenes.
        - RNN: Red nauronal recurrente, mayormente usada para capturar los efectos temporales en los datos.
    """)
    col = st.columns([1,1,1])
    image1 = Image.open('img/img_cnn.jpg')
    image2 = Image.open('img/rnn_img2.jpg')
    image3 = Image.open('img/convlstm_img.jpg')
    col[0].image(image1, caption='CNN')
    col[1].image(image2, caption='RNN')
    col[2].image(image3, caption='CONV-LSTM')

    st.markdown(
        """
        El objetivo de usar estos dos tipos de redes neuronales mediante la herramienta Tensorflow para hacer la predicción de los incidentes sobre una cuadricula ubicada sobre bogotá.
    """

)
with st.form("my_form2"):
    st.write("DEFINIR PARÁMETROS DE PREDICCIÓN")
    header1 = st.columns([1,1])
    row1 = st.columns([1,1])
    fecha_pred = row1[0].date_input("Define la fecha a predecir",min_value= dt.datetime(2022,1,1), max_value = dt.datetime(2022,12,31), value=dt.datetime(2022,7,15) )
    predecir = st.form_submit_button('PREDECIR')

if predecir:
    ###################### DATA
    #data = gpd.read_file("../../TRATAMIENTO DATA/outs/incidentes_sdm_2019_2023.geojson", encoding='latin-1')
    with st.spinner('Corriendo predicción...'):
        ## CARGAR EL MODELO
        #d = pd.read_csv(r"data\incidentes_sdm_2019_2023.csv")
        d = pd.read_feather("data/incidentes_sdm_2020_2023")
        q = d[['FECHA_x','LATITUDE','LONGITUDE']]
        q.columns = ['FECHA','LATITUD','LONGITUD']
        q['FECHA'] = pd.to_datetime(q['FECHA'])
        
        ruta_modelo = os.path.join(os.getcwd(),'models','last_model.keras')
        modelo = keras.models.load_model(ruta_modelo)
        
        fecha_min = fecha_pred - dt.timedelta(days=7)
        fecha_max = fecha_pred + dt.timedelta(days=6)
        data_pred = q[(q['FECHA'].dt.date >= fecha_min) & (q['FECHA'].dt.date <= fecha_max)]
        a,b = pred.predict_convlstm(data_pred,modelo)
        plot1, plot3, plot2 = st.columns((20,5,20))
        plot1.markdown("## INCIDENTES PREDICHOS")
        plot1.plotly_chart(a)
        plot2.markdown("## INCIDENTES REALES")
        plot2.plotly_chart(b)
    st.success('Done!')