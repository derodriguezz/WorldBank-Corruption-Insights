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

tab1, tab2, tab3 = st.tabs(["Correlacion", "Dispersion", "Matriz"])

with tab1:
   st.header("Grafica de Correlación entre variablessss")
   st.image("Procesamiento\graficas_sl\correlacion.png", width=900)

with tab2:
   st.header("Grafica de Dispersion entre variables")
   st.image("Procesamiento\graficas_sl\diagrama_dispersión.png", width=900)

with tab3:
   st.header("Tabla Matriz de Correlación")
   ruta_matriz_correlacion = r'Procesamiento\graficas_sl\matriz_correlacion.csv'
   matriz_correlacion = pd.read_csv(ruta_matriz_correlacion, index_col=0)
   # Aplicar estilos para resaltar valores
   estilos = matriz_correlacion.style.background_gradient(cmap='coolwarm').highlight_null('red')
   st.dataframe(estilos)
   # Mostrar la aplicación Streamlit
   st.write('Matriz de Correlación, entre variables de interes')

# Ruta del archivo Excel
file_path = "Extraccion/structured_data/reduced_df_normalized.xlsx"

# Cargar el archivo Excel en un DataFrame
df2 = pd.read_excel(file_path)

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
