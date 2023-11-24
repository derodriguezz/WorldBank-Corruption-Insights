import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# Cargar y mostrar la imagen
#imagen = cargar_imagen(ruta_imagen)
st.image('https://thelogisticsworld.com/wp-content/uploads/2023/09/Cepal.jpg', width=500)

# Título de la aplicación
st.title('CEPAL - INDICADORES ODS-CORR')

# Configurar la barra lateral con las pestañas
import streamlit as st

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlacion", "Matriz", "Dispersion" , "Dispersion Unificada" , "Modelo de Regresion", ])

with tab1:
   st.header("Grafica de Correlación entre variables")
   tab1.subheader("De acuerdo a la condición de normalidad de las variables analizadas, se realizan los análisis de correlación Spearman o Pearson según el resultado obtenido en cada una. En todos los análisis se determina la correlación entre la variable objetivo identificada y las variables de corrupción más relevantes identificadas a partir de la matriz de correlación.")
   st.image("Procesamiento/graficas_sl/correlacion.png", width=500)

with tab2:
   st.header("Tabla Matriz de Correlación")
   ruta_matriz_correlacion = r'Procesamiento/graficas_sl/matriz_correlacion.csv'
   matriz_correlacion = pd.read_csv(ruta_matriz_correlacion, index_col=0)
   # Aplicar estilos para resaltar valores
   estilos = matriz_correlacion.style.background_gradient(cmap='coolwarm').highlight_null('red')
   st.dataframe(estilos)
   # Mostrar la aplicación Streamlit
   st.write('Matriz de Correlación, entre variables de interes')


with tab3:
   st.header("Grafica de Dispersion entre variables")
   tab3.subheader ("")
   st.image("Procesamiento/graficas_sl/diagrama_dispersión.png", width=500)

with tab4:
   st.header("Grafica de Dispersion entre variables")
   tab4.subheader ("")
 

   
 
with tab5:
   st.header("Matriz Modelos de Regresion")
   tab5.subheader("Se implementa el entrenamiento de modelos de regresión lineal para prever la variable objetivo 'GDE_SL.GDP.PCAP.EM.KD'. En el primer escenario, se realiza la división de muestras sin considerar la variable categórica 'Country', utilizando 'CRP_CC.EST', 'CRP_GE.EST', 'CRP_RL.EST', 'CRP_VA.EST' como variables independientes. En los otros dos escenarios, se incorpora 'Country' y se generan modelos adicionales, uno con el conjunto completo de datos globales y otro con un conjunto reducido, evaluando así la influencia de esta variable en las predicciones. Se utiliza la técnica de regresión lineal y se evalúa el rendimiento del modelo en conjuntos de entrenamiento y prueba.")
   ruta_matriz_modelos = r'Procesamiento/graficas_sl/modelos_df.csv'
   matriz_modelos = pd.read_csv(ruta_matriz_correlacion, index_col=0)
   # Aplicar estilos para resaltar valores
   estilos = matriz_modelos.style.background_gradient(cmap='coolwarm').highlight_null('red')
   st.dataframe(estilos)
   # Mostrar la aplicación Streamlit
   st.write('Matriz modelos de regresion')




# Ruta del archivo Excel
file_path = "Extraccion/structured_data/reduced_df_normalized.xlsx"

# Cargar el archivo Excel en un DataFrame
df2 = pd.read_excel(file_path)

# Mostrar la tabla en Streamlit
#st.dataframe(df2)

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

