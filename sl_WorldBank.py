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
import streamlit as st

# Configurar la barra lateral con las pestañas
st.sidebar.title('Pestañas')
opciones = ['Pestaña 1', 'Pestaña 2', 'Pestaña 3']
seleccion = st.sidebar.radio('Ir a:', opciones)

# Contenido de las pestañas
if seleccion == 'Pestaña 1':
    st.subheader('Contenido de la Pestaña 1')

    # Agrega aquí el contenido específico de la pestaña 1

    # Agregar la imagen desde la ruta especificada
    ruta_imagen = r'C:/JHAA/CEPAL_3/WorldBank-Corruption-Insights/Procesamiento/graficas_sl/correlacion.png'
    st.image(ruta_imagen, caption='Matriz de Correlación', use_column_width=True)

elif seleccion == 'Pestaña 2':
    st.subheader('Contenido de la Pestaña 2')
    # Agrega aquí el contenido específico de la pestaña 2

elif seleccion == 'Pestaña 3':
    st.subheader('Contenido de la Pestaña 3')
    # Agrega aquí el contenido específico de la pestaña 3
