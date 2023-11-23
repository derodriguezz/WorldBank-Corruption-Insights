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

# Ruta del archivo Excel
file_path = "Extraccion/structured_data/reduced_df_normalized.xlsx"

# Cargar el archivo Excel en un DataFrame
df2 = pd.read_excel(file_path)

# Mostrar la tabla en Streamlit
st.dataframe(df2)

#df2 = pd.read_excel(file_path_2)

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
