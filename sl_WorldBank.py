import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
from PIL import Image
import os

# Título de la aplicación
st.title('CEPAL - INDICADORES ODS-CORR')

# Cargar y mostrar la imagen
#imagen = cargar_imagen(ruta_imagen)
st.image('https://thelogisticsworld.com/wp-content/uploads/2023/09/Cepal.jpg')



# Configurar la barra lateral con las pestañas
st.sidebar.title('Pestañas')
opciones = ['Pestaña 1', 'Pestaña 2', 'Pestaña 3']
seleccion = st.sidebar.radio('Ir a:', opciones)

# Contenido de las pestañas
if seleccion == 'Pestaña 1':
    st.subheader('Contenido de la Pestaña 1')
    # Agrega aquí el contenido específico de la pestaña 1

# Cargar el DataFrame desde el archivo Excel
    archivo_excel = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\reduced_df_normalized.xlsx'
    df = pd.read_excel(archivo_excel)

# Seleccionar las columnas que comienzan con "CRP" y "GDE"
    columnas_crp = [col for col in df.columns if col.startswith('CRP')]
    columnas_gde = [col for col in df.columns if col.startswith('GDE')]

# Crear un nuevo DataFrame con las columnas seleccionadas
    df_corr = df[columnas_crp + columnas_gde]

# Calcular la matriz de correlación
    matriz_correlacion = df_corr.corr()

# Configurar el estilo del gráfico
    sns.set(style='white')

# Crear un mapa de calor de la matriz de correlación
    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

# Mostrar el gráfico
    plt.title('Matriz de Correlación entre Variables CRP y GDE')
    plt.show()


elif seleccion == 'Pestaña 2':
    st.subheader('Contenido de la Pestaña 2')
    # Agrega aquí el contenido específico de la pestaña 2

elif seleccion == 'Pestaña 3':
    st.subheader('Contenido de la Pestaña 3')
    # Agrega aquí el contenido específico de la pestaña 3

