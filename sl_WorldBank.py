import streamlit as st
import pandas as pd
import plotly.express as px


# Título de la aplicación
st.title('CEPAL - ANALISIS DE INDICADORES')

# Cargar una imagen desde un archivo local
image_path = 'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\CEPAL.jpg'  # Reemplaza con la ruta a tu imagen
image = st.image(image_path, caption='Descripción opcional', use_column_width=True)

# También puedes cargar una imagen directamente desde una URL
# image_url = 'https://ejemplo.com/ejemplo.jpg'
# image = st.image(image_url, caption='Descripción opcional', use_column_width=True)

# Puedes ajustar el tamaño de la imagen y agregar texto debajo
 st.image(image_path, caption='Otra descripción', width=300, height=200)

# Puedes usar Markdown para formatear el texto
# st.image(image_path, caption='**Texto en negrita**', use_column_width=True)

# También puedes cambiar la posición del texto (abajo, arriba, izquierda, derecha)
# st.image(image_path, caption='Texto a la izquierda', use_column_width=True, location='left')


# Cargar los datos
file_path_1 = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\merged_df_normalized.xlsx'
file_path_2 = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\reduced_df_normalized.xlsx'

df1 = pd.read_excel(file_path_1)
df2 = pd.read_excel(file_path_2)

# Crear una lista de dataframes y nombres
dfs = [df1, df2]
df_names = ['merged_df', 'reduced_df']

# Streamlit app
st.title('ANALISIS DE DATOS - CEPAL')

# Selector de archivo
selected_df = st.selectbox('Seleccionar Archivo:', df_names)

# Selector de variables
selected_variables = st.multiselect('Seleccionar Variable(s):', df1.columns)

# Filtro por país
selected_countries = st.multiselect('Seleccionar País(es):', df1['Country'].unique())

# Filtro por año
selected_year = st.slider('Seleccionar Año:', min_value=df1['Year'].min(), max_value=df1['Year'].max(), value=(df1['Year'].min(), df1['Year'].max()))

# Filtrar el DataFrame
filtered_df = dfs[df_names.index(selected_df)]
filtered_df = filtered_df[(filtered_df['Country'].isin(selected_countries)) & (filtered_df['Year'] >= selected_year[0]) & (filtered_df['Year'] <= selected_year[1])]

# Aplicar filtro de variables seleccionadas
if selected_variables:
    filtered_df = filtered_df[selected_variables + ['Year', 'Country']]

# Graficar con Plotly Express
fig = px.line(filtered_df, x='Year', y=selected_variables, color='Country', title='Gráfica Interactiva')
st.plotly_chart(fig)

# Mostrar tabla de datos
st.dataframe(filtered_df)
