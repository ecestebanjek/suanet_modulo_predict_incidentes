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
import oracledb
import sqlalchemy as sa
from sqlalchemy import select, and_, func

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

########CONSULTA BASE DE DATOS
@st.cache_data
def consult_data(fecha):
    fecha_inicial0 = fecha - dt.timedelta(days=7)
    fecha_final0 = fecha #+ dt.timedelta(days=7)
    fecha_inicial = fecha_inicial0.strftime('%Y-%m-%d %H:%M:%S')
    fecha_final = fecha_final0.strftime('%Y-%m-%d %H:%M:%S')
    ## CONEXIÓN A LA BASE DE DATOS
    dialect = 'oracle'
    sql_driver = 'oracledb'
    ## ORACLE SDM ## hacer esto con variables de entorno
    un = os.environ["UNSN"]
    host = os.environ["HOST"]
    port = os.environ["PORT"]
    sn = os.environ["UNSN"]
    pw = os.environ["P"]
    # try:
    try:
        to_engine: str = dialect + '+' + sql_driver + '://' + un + ':' + pw + '@' + host + ':' + str(port) + '/?service_name=' + sn
        connection = sa.create_engine(to_engine)
        query = f"SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT WHERE INCIDENTDATE BETWEEN TO_TIMESTAMP('{fecha_inicial}', 'YYYY-MM-DD HH24:MI:SS') AND TO_TIMESTAMP('{fecha_final}', 'YYYY-MM-DD HH24:MI:SS')"
        # test_df = pd.read_sql_query('SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT WHERE INCIDENTDATE BETWEEN TO_TIMESTAMP("' + str(fecha_inicial) + '") AND TO_TIMESTAMP("'+ str(fecha_final)+'"), "YYYY-MM-DD HH24:MI:SS"', connection)
        test_df = pd.read_sql_query(query, connection)
        ## Selección especifica de dias
        test_df = test_df[test_df['latitude']!=0]
        test_df = test_df[['incidentdate','latitude','longitude']]
        st.info('Base de datos consultada - '+str(test_df.shape[0]) + " incidentes consultados entre " + str(test_df.incidentdate.min()) + " y " + str(test_df.incidentdate.max()))
        test_df.columns = ['FECHA','LATITUD','LONGITUD']
        test_df['FECHA'] = pd.to_datetime(test_df['FECHA'])
        return test_df
    except:
        d = pd.read_feather("data/incidentes_sdm_2020_2023")
        q = d[['FECHA_x','LATITUDE','LONGITUDE']]
        q.columns = ['FECHA','LATITUD','LONGITUD']
        q['FECHA'] = pd.to_datetime(q['FECHA'])
        fecha_min = fecha - dt.timedelta(days=7)
        # fecha_max = fecha #+ dt.timedelta(days=6)
        fecha_max = fecha + dt.timedelta(days=6)
        data_pred = q[(q['FECHA'].dt.date >= fecha_min) & (q['FECHA'].dt.date <= fecha_max)]
        #st.info('Base de datos consultada - '+str(data_pred.shape[0]) + " incidentes consultados entre " + str(data_pred.FECHA.min()) + " y " + str(data_pred.FECHA.max()))
        return data_pred

################################# PAGINA DE INICIO
## Show in webpage
st.markdown("# MODELO PREDICTIVO DE INCIDENTES")

# with st.expander("Ver explicación"):
#     st.markdown(
#         """
#         # MODELO PREDICTIVO DE INCIDENTES
        
#         ## Explicación
        
#         Se plantea un modelo que mezcla dos tipos de redes neuronales:
#         - CNN: Red neuronal convolucional, mayormente usada para capturar los efectos espaciales de los datos y usadas principalmente para el análisis de imágenes.
#         - RNN: Red nauronal recurrente, mayormente usada para capturar los efectos temporales en los datos.
#     """)
#     col = st.columns([1,1,1])
#     image1 = Image.open('img/img_cnn.jpg')
#     image2 = Image.open('img/rnn_img2.jpg')
#     image3 = Image.open('img/convlstm_img.jpg')
#     col[0].image(image1, caption='CNN')
#     col[1].image(image2, caption='RNN')
#     col[2].image(image3, caption='CONV-LSTM')

#     st.markdown(
#         """
#         El objetivo de usar estos dos tipos de redes neuronales mediante la herramienta Tensorflow para hacer la predicción de los incidentes sobre una cuadricula ubicada sobre bogotá.
#     """

# )
with st.form("my_form2"):
    st.write("DEFINIR PARÁMETROS DE PREDICCIÓN")
    header1 = st.columns([1,1])
    row1 = st.columns([1,1])
    fecha_pred = row1[0].date_input("Define la fecha a predecir",min_value= dt.datetime(2024,1,1), max_value = dt.date.today() + dt.timedelta(days=1), value=dt.date.today() + dt.timedelta(days=1) )
    predecir = st.form_submit_button('PREDECIR')

plot3, plot1, plot2 = st.columns((1,30,1))
map = st.empty()
fig2 = px.choropleth_mapbox(pd.DataFrame(),height = 1000, width=1000,
                        #geojson=grid.geometry,
                        #locations=grid.index,
                        color_continuous_scale="reds",
                        #color="y",
                        center={"lat": 4.653,  "lon": -74.1},
                        mapbox_style="open-street-map",#"open-street-map",Cartodb dark_matter ,"carto-positron"
                        opacity=0.5,
                        zoom=10)
with plot1:
    map.plotly_chart(fig2, use_container_width=True)

if predecir:
    ###################### DATA
    #data = gpd.read_file("../../TRATAMIENTO DATA/outs/incidentes_sdm_2019_2023.geojson", encoding='latin-1')
    with st.spinner('Corriendo predicción...'):
        ## CARGAR EL MODELO
        #d = pd.read_csv(r"data\incidentes_sdm_2019_2023.csv")
        # d = pd.read_feather("data/incidentes_sdm_2020_2023")
        # q = d[['FECHA_x','LATITUDE','LONGITUDE']]
        # q.columns = ['FECHA','LATITUD','LONGITUD']
        # q['FECHA'] = pd.to_datetime(q['FECHA'])
        data_pred = consult_data(fecha_pred)
        if data_pred.shape[0]>1:
            ruta_modelo = os.path.join('models','last_model_2.h5')
            modelo = keras.models.load_model(ruta_modelo)
            
            a,b = pred.predict_convlstm(data_pred,modelo)
            # plot3, plot1, plot2 = st.columns((1,30,1))
            with plot1:
                st.markdown("#### INCIDENTES PREDICHOS")
                map.plotly_chart(a, use_container_width=True)
        else:
            st.error('No se encontraron datos')
        
        # plot1.plotly_chart(a, use_container_width=True)
        # plot2.markdown("## INCIDENTES REALES")
        # plot2.plotly_chart(b)
    st.success('Done!')