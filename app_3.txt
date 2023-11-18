import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Ruta a la imagen
ruta_imagen = "C:\JHAA\CEPAL_2\CEPAL.jpg"


# Función para cargar y mostrar la imagen
def cargar_imagen(ruta):
    imagen = Image.open(ruta)
    return imagen

# Título de la aplicación
st.title('CEPAL - INDICADORES ODS-CORR')

# Cargar y mostrar la imagen
imagen = cargar_imagen(ruta_imagen)
st.image(imagen, caption='Imagen "CEPAL"', use_column_width=True)



import pandas as pd
import streamlit as st
import plotly.express as px

# Cargar datos desde el archivo Excel
file_path = r'C:\JHAA\CEPAL_2\archivo_reduced_df.xlsx'
df = pd.read_excel(file_path)

# Interfaz de usuario con filtros desplegables
st.sidebar.title('Filtros')
selected_countries = st.sidebar.multiselect('Seleccionar países', df['Country'].unique())
selected_columns = st.sidebar.multiselect('Seleccionar columnas', df.columns)

# Filtrar el DataFrame según los países seleccionados y las columnas seleccionadas
filtered_df = df[df['Country'].isin(selected_countries)]
filtered_df = filtered_df[['Year', 'Country'] + selected_columns]  # Añadir la columna 'Year' y 'Country'

# Crear gráfico interactivo
#fig = px.line(filtered_df, x='Year', y=selected_columns, color='Country')
fig = px.bar(filtered_df, x='Year', y=selected_columns, color='Country', barmode='stack')
#fig = px.area(filtered_df, x='Year', y=selected_columns, color='Country')
#fig = px.scatter(filtered_df, x='Year', y=selected_columns, color='Country')
#fig = px.violin(filtered_df, x='Year', y=selected_columns, color='Country', box=True, points="all")



# Mostrar gráfico en la aplicación de Streamlit
st.title('RELACION INDICADORES ENTRE PAISES')
st.plotly_chart(fig)




import pandas as pd
import streamlit as st
import plotly.express as px

# Ruta del archivo
file_path = r'C:\JHAA\CEPAL_2\archivo_reduced_df2.xlsx'

# Función para cargar datos desde el archivo
def load_data():
    return pd.read_excel(file_path)

# Cargar datos
df = load_data()

# Interfaz de usuario con filtros desplegables
st.sidebar.title('Filtros - COLOMBIA')
selected_columns = st.sidebar.multiselect('Seleccionar Indicador(es)', df.filter(regex='^(CRP|GDE)').columns)

# Añadir la columna 'Country' con valor fijo
df['Country'] = 'Colombia'
filtered_df = df[['Year', 'Country'] + selected_columns]  # Añadir la columna 'Year' y 'Country'

# Crear gráfico interactivo
#fig = px.line(filtered_df, x='Year', y=selected_columns, color='Country')
fig = px.area(filtered_df, x='Year', y=selected_columns, color='Country')
# Mostrar gráfico en la aplicación de Streamlit
st.title('INDICADORES - PARA COLOMBIA')
st.plotly_chart(fig)
