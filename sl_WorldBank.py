import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
from PIL import Image
import os

# Convertir la imagen a formato PNG
##imagen_jpg = 'cepal.jpg'
##imagen_png = 'cepal.png'

##Image.open(imagen_jpg).save(imagen_png)

# Mostrar la imagen convertida
##imagen = Image.open(imagen_png)
#imagen.show()

# Ruta a la imagen
###ruta_imagen = 'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\cepal.png'  # Esta es la ruta relativa al directorio de la aplicación en Streamlit Community Cloud

# Función para cargar y mostrar la imagen
##def cargar_imagen(ruta):
##    imagen = Image.open(ruta)
#    return imagen

# Título de la aplicación
st.title('CEPAL - INDICADORES ODS-CORR')

# Cargar y mostrar la imagen
#imagen = cargar_imagen(ruta_imagen)
st.image('https://thelogisticsworld.com/wp-content/uploads/2023/09/Cepal.jpg')

# Resto del código...

# Cargar los datos
file_path_2 = r'C:\\JHAA\\CEPAL_3\\WorldBank-Corruption-Insights\\Extraccion\\structured_data\\reduced_df_normalized.xlsx'
df2 = pd.read_excel(file_path_2)

# Streamlit app
st.title('ANALISIS DE DATOS - CEPAL')

# Selector de variables
selected_variables = st.multiselect('Seleccionar Variable(s):', df2.columns)

# Filtro por país
selected_countries = st.multiselect('Seleccionar País(es):', df2['Country'].unique())

# Filtro por año
selected_year = st.slider('Seleccionar Año:', min_value=df2['Year'].min(), max_value=df2['Year'].max(), value=(df2['Year'].min(), df2['Year'].max()))

# Filtrar el DataFrame
filtered_df = df2[(df2['Country'].isin(selected_countries)) & (df2['Year'] >= selected_year[0]) & (df2['Year'] <= selected_year[1])]

# Aplicar filtro de variables seleccionadas
if selected_variables:
    filtered_df = filtered_df[selected_variables + ['Year', 'Country']]

# Graficar con Plotly Express
fig = px.line(filtered_df, x='Year', y=selected_variables, color='Country', title='Gráfica Interactiva')
st.plotly_chart(fig)

# Mostrar tabla de datos
#st.dataframe(filtered_df)
