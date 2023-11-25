import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# Cargar y mostrar la imagen
#imagen = cargar_imagen(ruta_imagen)
st.image('https://thelogisticsworld.com/wp-content/uploads/2023/09/Cepal.jpg', width=900)

# Título de la aplicación
st.title('CEPAL - INDICADORES ODS-CORR')

# Configurar la barra lateral con las pestañas
import streamlit as st

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Correlacion", "Matriz de Correlación", "Dispersion" , "Resultados Modelos de Regresión" , "Modelo de Regresion", "Gráfica iteractiva"])

with tab1:
   st.header("Grafica de Correlación entre variables")
   tab1.subheader("De acuerdo a la condición de normalidad de las variables analizadas, se realizan los análisis de correlación Spearman o Pearson según el resultado obtenido en cada una. En todos los análisis se determina la correlación entre la variable objetivo identificada y las variables de corrupción más relevantes identificadas a partir de la matriz de correlación.")
   st.image("Procesamiento/graficas_sl/correlacion.png", width=700)

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
   st.image("Procesamiento/graficas_sl/diagrama_dispersión.png", width=900)

with tab4:
   st.header("Resultados modelos de Regresión")
   st.write ("En este análisis, hemos utilizado una serie de indicadores del Banco Mundial que se centran en la gobernanza y el desempeño de los gobiernos.")  
   tab4.subheader ("Control de la Corrupción (CC.EST):")
   st.write ("Control de la Corrupción evalúa la percepción de hasta qué punto el poder público se ejerce para beneficio privado, abarcando formas tanto menores como mayores de corrupción.")
   tab4.subheader ("Efectividad del Gobierno (GE.EST):")
   st.write  ("Efectividad del Gobierno mide la calidad de los servicios públicos, la independencia del servicio civil y la credibilidad del compromiso del gobierno con sus políticas.")
   tab4.subheader ("Estado de Derecho - Cumplimiento de la ley (RL.EST):")
   st.write  ("Estado de Derecho refleja la confianza y el cumplimiento de las reglas de la sociedad, incluyendo la aplicación de contratos, derechos de propiedad, la actuación policial y judicial, así como la probabilidad de crimen y violencia.")
   tab4.subheader ("Voz y Rendición de Cuentas (VA.ESTb4):")
   st.write ("Voz y Rendición de Cuentas evalúa la participación ciudadana en la selección del gobierno y la libertad de expresión, asociación y medios de comunicación.")
   st.header("Grafica Modelos aplicados")
   tab4.subheader ("Gráfica de Resultados")
   st.image("Procesamiento/graficas_sl/modelo2_rf_sp.png", width=900)
   st.image("Procesamiento/graficas_sl/modelo2_rf_cp.png", width=900)
   st.image("Procesamiento/graficas_sl/modelo2_cp_pred.png", width=900)
   st.image("Procesamiento/graficas_sl/modelo2_cp_test.png", width=900)
   st.image("Procesamiento/graficas_sl/modelo2_rf_sp.png", width=900)
   st.image("Procesamiento/graficas_sl/modelo2_rf_cp.png", width=900)
   

with tab5:
   st.header("Matriz Modelos de Regresion")
   tab5.subheader("Resultado - Tabla comparativa entre modelos empleados")
   ruta_matriz_modelos = r'Procesamiento/graficas_sl/modelos_df.csv'
   matriz_modelos = pd.read_csv(ruta_matriz_modelos, index_col=0)
   # Aplicar estilos para resaltar valores
   estilos = matriz_modelos.style.background_gradient(cmap='coolwarm').highlight_null('red')
   st.dataframe(matriz_modelos)
   # Mostrar la aplicación Streamlit
   st.write('Matriz modelos de regresion')
   tab5.subheader ("Análisis de componentes principales")
   st.write("El hecho de que con 2 o 3 componentes principales se explique más del 95% de la varianza sugiere que estos componentes capturan la mayoría de la información de las variables de corrupción. El gráfico de varianza explicada acumulativa es útil para determinar cuántos componentes son necesarios para conservar una cantidad significativa de varianza.")
   st.write("El hecho de observar dos clusters distintos en el gráfico de resultados del PCA sugiere que estos clusters se deban a patrones o estructuras que los componentes principales han identificado. Podrían representar grupos o tendencias específicas.")
   st.write("Los pesos de los componentes principales indican la contribución de cada variable original a los componentes principales. En tu caso, el primer componente principal (PC1) tiene pesos relativamente altos para todas las variables, indicando que está capturando información general de todas ellas. Por otro lado, el segundo componente principal (PC2) tiene un peso significativamente alto para la variable 'CRP_GE.EST' y un peso negativo para 'CRP_VA.EST', lo que sugiere que PC2 podría estar relacionado con variaciones específicas en estas dos variables.")
   st.write("Los scores de los componentes principales representan las proyecciones de los datos originales en el espacio de los componentes principales. Los valores más altos o más bajos en los scores indican la posición relativa de cada observación en el espacio de los componentes principales:")
   tab5.subheader("Componente Principal 1 (PC1):")
   st.write("Este componente parece capturar una tendencia general o patrón común en todas las variables.")
   tab5.subheader("Componente Principal 2 (PC2)")
   st.write("Este componente parece capturar variaciones específicas relacionadas con 'CRP_GE.EST' y 'CRP_VA.EST'.")
   st.image("Procesamiento/graficas_sl/varianza_explicada_acumulativa.png", width=900)
   st.image("Procesamiento\graficas_sl\Resultados_pca.png", width=900)

with tab6:
   st.header("Grafica Iteractiva -Indicadores - Pais - Rango de tiempo")
   tab6.subheader("Comportamiento de los indicadores por Pais - CELAP ")
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

