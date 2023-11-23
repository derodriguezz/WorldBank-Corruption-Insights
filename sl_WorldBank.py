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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la barra lateral con las pestañas
st.sidebar.title('Pestañas')
opciones = ['Pestaña 1', 'Pestaña 2', 'Pestaña 3']
seleccion = st.sidebar.radio('Ir a:', opciones)

# Cargar datos desde el archivo
ruta_archivo = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\reduced_df_normalized.xlsx'
df = pd.read_excel(ruta_archivo)

# Contenido de las pestañas
if seleccion == 'Pestaña 1':
    st.subheader('Contenido de la Pestaña 1')

    # Agregar aquí el contenido específico de la pestaña 1
    st.write("Gráfico de barras:")
    columna_seleccionada = st.selectbox("Seleccione una columna:", df.columns)
    
    # Crear un gráfico de barras
    fig, ax = plt.subplots()
    sns.countplot(x=columna_seleccionada, data=df, ax=ax)
    st.pyplot(fig)

elif seleccion == 'Pestaña 2':
    st.subheader('Contenido de la Pestaña 2')
    # Agregar aquí el contenido específico de la pestaña 2

elif seleccion == 'Pestaña 3':
    st.subheader('Contenido de la Pestaña 3')
    # Agregar aquí el contenido específico de la pestaña 3
