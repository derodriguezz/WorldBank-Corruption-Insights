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
#st.plotly_chart(fig)

# Mostrar tabla de datos
#st.dataframe(filtered_df)


# Crear una lista de opciones para las pestañas
########################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table
from openpyxl import load_workbook

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

# Obtener la ubicación de la celda donde se insertará el gráfico
hoja_destino = 'Pestaña1'  # Cambia esto según el nombre de la pestaña deseada
ubicacion_celda = 'A1'

# Guardar el gráfico en el archivo Excel
with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
    writer.book = load_workbook(archivo_excel)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)

    # Guardar el gráfico en la hoja de Excel
    imagen = plt.imshow([[0, 0], [0, 0]], cmap='coolwarm')
    plt.gca().set_visible(False)
    tabla = table(plt.gca(), matriz_correlacion, loc='center', colWidths=[0.1] * len(matriz_correlacion.columns))
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)

    writer.sheets[hoja_destino].add_image(imagen, ubicacion_celda)
    plt.close()

    # Guardar el archivo Excel con el gráfico
    writer.save()
