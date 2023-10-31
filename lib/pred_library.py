import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GRU, Concatenate,  Dropout, Masking
import keras
from keras import layers
import os
import math
import shapefile as shp
from sklearn.preprocessing import MinMaxScaler
import folium
from folium import plugins
from IPython.display import display, clear_output
from folium.features import Choropleth
import plotly.express as px

########### FUNCIONES
## Funciones
## Funciones

def prepare_from_satialtimedf(a, gr = False):
    df = a.copy()
    '''Recibe un df con fecha-latitud-longitud'''
    #### Creando las caracteristicas de tiempo
    df['WEEK_DAY'] = df['FECHA'].dt.dayofweek
    df['MONTH_DAY'] = df['FECHA'].dt.days_in_month
    df['YEAR_DAY'] = df['FECHA'].dt.dayofyear
    df['MONTH'] = df['FECHA'].dt.month
    df['WEEK_END'] = np.where(df['WEEK_DAY'].isin([5,6]) ,1,0)
    
    
    columns_to_normalize = ['WEEK_DAY', 'MONTH_DAY', 'YEAR_DAY','MONTH','WEEK_END']
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    df.sort_values(by=['FECHA'], inplace=True)
    
    #### Convierte en gdf
    df = gpd.GeoDataFrame(df, crs="EPSG:4326",geometry=gpd.points_from_xy(x=df.LONGITUD, y=df.LATITUD) )
    
    #### Crea la red
    minx, miny, maxx, maxy = df.total_bounds
    if gr:
        grid, nx, ny = create_grid(0.005,0.005, maxx, maxy, minx, miny)
    else:
        grid = gpd.read_file(os.path.join("data","cuadricula_predicciones.shp"))
        grid = grid.set_crs("EPSG:4326")
        grid['g'] = grid.geometry
        dx = 0.005
        dy = 0.005
        nx = 42#int(math.ceil(abs(maxx - minx)/dx))
        ny = 72#int(math.ceil(abs(maxy - miny)/dy))
    #### Pega posiciones de la red a df
    df = df.sjoin(grid, how='left',predicate ='within')
    df.dropna(subset='ID', inplace=True)
    import ast
    df['z'] = df['ID'].apply(ast.literal_eval)
    df['y'] = df['z'].apply(lambda g:g[0])
    df['x'] = df['z'].apply(lambda g:g[1])
    df.drop('z', axis=1, inplace=True)
    
    #### Cluster by grid by day
    df.reset_index(inplace=True)
    df = df.groupby(['FECHA','ID']).agg({'index':'count',
                        'x' : 'first',
                        'y' : 'first',
                        'WEEK_DAY':'first',
                        'YEAR_DAY':'first',
                        'MONTH':'first',
                        'MONTH_DAY':'first',
                        'WEEK_END':'first'}).reset_index()
    df.columns = ['FECHA', 'ID', 'COUNTS', 'x', 'y', 'WEEK_DAY', 'YEAR_DAY', 'MONTH','MONTH_DAY', 'WEEK_END']
    # df.columns = []
    
    ## Maximizar atención en counts
    ##df['COUNTS'] = df['COUNTS']**1.5 # Viendo si funciona
    
    return df,nx, ny
def extract_day_matrix(df, nx, ny):
    df = df.reset_index()
    matrix_day = np.zeros((nx, ny,6))
    for a in range(nx):
        for b in range(ny):
            matrix_day[a][b][1] = df.loc[0,'WEEK_DAY']
            matrix_day[a][b][2] = df.loc[0,'YEAR_DAY']
            matrix_day[a][b][3] = df.loc[0,'MONTH']
            matrix_day[a][b][4] = df.loc[0,'MONTH_DAY']
            matrix_day[a][b][5] = df.loc[0,'WEEK_END']
    
    for i in range(len(df)):
        x = df.loc[i,'x']-1
        y = df.loc[i,'y']-1
        matrix_day[y][x][0] = df.loc[i,'COUNTS']
    return matrix_day



def crear_window_tensor(matriz_tot, batch):
    dataset = tf.convert_to_tensor(matriz_tot)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(14, shift=7, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(14))
    dataset = dataset.map(lambda window: (window[:-7], window[7:,:,:,0:1]))
    dataset = dataset.batch(batch).prefetch(1)
    return dataset

def crear_window_tensor_pred(matriz_tot, batch):
    dataset = tf.convert_to_tensor(matriz_tot)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(14, shift=7, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(14))
    #dataset = dataset.window(7, drop_remainder=True)
    #dataset = dataset.flat_map(lambda window: window.batch(7))
    #dataset = dataset.map(lambda window: (window, window[7:,:,:,0:1]))
    dataset = dataset.map(lambda window: (window[:-7], window[7:,:,:,0:1]))
    dataset = dataset.batch(batch)
    return dataset

### Actualizar función

def predict_convlstm(a,modelo):
    b = a.copy()
    ### Prepara la data inicial
    df,nx, ny = prepare_from_satialtimedf(b)
    ### Convierte a tensores
    matriz_tot = []
    fechas = df['FECHA'].unique()
    for date in fechas:
        a = df[df['FECHA']== date].reset_index()
        matriz_tot.append(extract_day_matrix(a,nx, ny))
        
    ### Crea tensor_windows
    t_test = crear_window_tensor_pred(matriz_tot,1)
    
    # ### Crea el modelo
    # ### Imprime las imagenes reales y predichas de una predicción
    y_pred = modelo.predict(t_test)
    y_pred = tf.squeeze(y_pred[0,0:1,:,:,0]).numpy()
    for x,y in t_test:
        y = tf.squeeze(y[0,0:1,:,:,0]).numpy()
        break
    ## Extrayendo primera predicción de y_pred
    tuplas = []
    valores = []
    for idx, x in np.ndenumerate(y_pred):
        tuplas.append(idx)
        valores.append(x)
        
    y_pred = pd.DataFrame([tuplas, valores]).T
    y_pred.columns = ['ID','y_pred']
    y_pred['ID'] = y_pred['ID'].astype(str)
    y_pred['ID'] = y_pred['ID'].str.replace(' ','')
    
    
    ## Estrayendo primer y real
    tuplas = []
    valores = []
    for idx, x in np.ndenumerate(y):
        tuplas.append(idx)
        valores.append(x)
        
    y = pd.DataFrame([tuplas, valores]).T
    y.columns = ['ID','y']
    y['ID'] = y['ID'].astype(str).str.replace(' ','')
    
    grid = gpd.read_file(os.path.join("data","cuadricula_predicciones.shp"))
    grid = grid.set_crs("WGS 84")
    grid['g'] = grid.geometry
    grid['ID'] = grid['ID'].str.replace(' ','')
    
    grid = grid.merge(y_pred, how='left', on='ID')
    grid = grid.merge(y, how='left', on='ID')
    grid['y_pred'] = pd.to_numeric(grid['y_pred'])
    grid['y'] = pd.to_numeric(grid['y'])
    
    # Se quita la importancia agregada
    ##grid['y_pred'] = grid['y_pred']**(1/1.5)
    ##grid['y'] = grid['y']**(1/1.5)
    #Se reescala y redondea
    # max_y = np.max(grid['y'])
    # max_yp = np.max(grid['y_pred'])
    # grid['y_pred'] = grid['y_pred']*max_y/max_yp
    # grid['y_pred'] = np.where(grid['y_pred']<1,0,grid['y_pred'])
    grid.fillna(0, inplace=True)
    #grid.dropna(subset="y_pred", inplace=True)

    # grid.plot(column="y_pred")
    # grid.plot(column="y")

    ## Choroplet
    fig1 = px.choropleth_mapbox(grid,height =  800, width=700,
                           geojson=grid.geometry,
                           locations=grid.index,
                           color_continuous_scale="reds",
                           color="y_pred",
                           center={"lat": 4.653,  "lon": -74.1},
                           mapbox_style="open-street-map",#"open-street-map",Cartodb dark_matter ,"carto-positron"
                           opacity=0.5,
                           zoom=10)
    fig1 = fig1.update_traces(
        marker_line_width=0
    )
    #fig1.show()
    
    fig2 = px.choropleth_mapbox(grid,height = 800, width=700,
                           geojson=grid.geometry,
                           locations=grid.index,
                           color_continuous_scale="reds",
                           color="y",
                           center={"lat": 4.653,  "lon": -74.1},
                           mapbox_style="open-street-map",#"open-street-map",Cartodb dark_matter ,"carto-positron"
                           opacity=0.5,
                           zoom=10)
    fig2 = fig2.update_traces(
        marker_line_width=0
    )
    #fig2.show()
    return fig1, fig2