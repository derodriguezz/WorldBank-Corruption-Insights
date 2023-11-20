import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar los datos
file_path_1 = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\merged_df_normalized.xlsx'
file_path_2 = r'C:\JHAA\CEPAL_3\WorldBank-Corruption-Insights\Extraccion\structured_data\reduced_df_normalized.xlsx'

df1 = pd.read_excel(file_path_1)
df2 = pd.read_excel(file_path_2)

# Crear una lista de dataframes y nombres
dfs = [df1, df2]
df_names = ['merged_df', 'reduced_df']

# Streamlit app
st.title('Análisis de Datos')

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
