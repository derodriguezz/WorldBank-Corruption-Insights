import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# Cargar y mostrar la imagen
#imagen = cargar_imagen(ruta_imagen)
st.image('https://thelogisticsworld.com/wp-content/uploads/2023/09/Cepal.jpg',  width=600 ) 


# Título de la aplicación
st.title('Corrupción y Desarrollo Económico en los Paises de la CEPAL')

# Configurar la barra lateral con las pestañas
import streamlit as st

tab1, tab2, tab3 = st.tabs(["INTRODUCCION ", "RESULTADOS RELACION ENTRE VARIABLES", "EXPLORA LOS DATOS"])
with tab1:
   st.header("BIENVENIDO")
   st.write("Esta aplicación web se presenta como una valiosa herramienta para comprender la relación entre la corrupción y el desarrollo económico. La información que se presenta parte de un proyecto de investigación respaldado por metodologías de Big Data, las cuales analizan los indicadores más significativos para modelar esta relación. Como sustento, se exponen los resultados del estudio, que comprenden análisis de correlación, modelos de regresión y evaluaciones de componentes principales.")
   st.write("El propósito subyacente de este proyecto es robustecer los procesos de observación, control e intervención relacionados con la corrupción. Se concibe como un recurso valioso, sumándose a las iniciativas destinadas a mitigar los efectos adversos de la corrupción en el desarrollo económico. Selecciona cualquiera de las secciones disponibles para explorar a fondo los resultados o para adentrarte en el análisis detallado de los datos.")
   
   st.header("RESUMEN")
   st.write("En este análisis, empleamos diversos indicadores del Banco Mundial que se centran en la gobernanza y el desempeño gubernamental.")
   st.write("Control de la Corrupción (CC.EST): Evalúa la percepción de cómo el poder público se ejerce para beneficio privado, abarcando diversas formas de corrupción.")
   st.write("Efectividad del Gobierno (GE.EST): Mide la calidad de los servicios públicos, la independencia del servicio civil y la credibilidad del compromiso gubernamental con sus políticas.")
   st.write("Estado de Derecho - Cumplimiento de la ley (RL.EST): Refleja la confianza y el cumplimiento de las reglas de la sociedad, incluyendo la aplicación de contratos, derechos de propiedad, la actuación policial y judicial, así como la probabilidad de crimen y violencia")
   st.write("Voz y Rendición de Cuentas (VA.EST): Evalúa la participación ciudadana en la selección del gobierno y la libertad de expresión, asociación y medios de comunicación.")
   st.write("Al realizar un modelo de regresión lineal con estos indicadores, se identificaron relaciones significativas entre las variables independientes y el Producto Interno Bruto (PIB) per cápita ajustado por paridad de poder adquisitivo (SL.GDP.PCAP.EM.KD).")
   st.write("A pesar de que se encontraron fuertes correlaciones entre las variables para el caso de Colombia, es crucial señalar que el modelo no presenta un alto nivel predictivo para estos datos, posiblemente debido a factores únicos en el contexto colombiano no completamente capturados por los indicadores utilizados. A pesar de esta limitación, las fuertes correlaciones entre los indicadores de gobernanza y el PIB per cápita sugieren que mejoras en el control de la corrupción, la efectividad gubernamental, el estado de derecho y la voz ciudadana podrían tener implicaciones positivas en el desarrollo económico de un país.")

with tab3:
   st.write("En esta sección puedes elegir los parámetros de Pais, Indicador y Año, para conocer al detalle el comportamiento de los indicadores tratados para los países en un margen de tiempo")
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

with tab2:
   #st.header("RESULTADOS")
   tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dispersiòn", "Normalidad","Correlaciòn" , "Resultados Modelos de Regresión" , "Análisis Componentes Principales"])
   with tab3:
      st.header("Resultados - Análisis de correlación")
      st.write("Los indicadores de corrupción son definidos como variables independientes, buscando sus relaciones más fuertes con aquellos que forman parte de los Objetivos de Desarrollo Global y de Desarrollo Sostenible (ODS #8). Mediante una matriz de correlación, identificamos que, entre todos los indicadores evaluados, el Producto Interno Bruto (PIB) per cápita ajustado por paridad de poder adquisitivo (SL.GDP.PCAP.EM.KD) mantiene una relación positiva y sólida con varios indicadores de corrupción. Esta conclusión se valida al analizar el conjunto de datos específico para Colombia.")
      st.write('Matriz de Correlación, entre variables de interes')
      st.image("Procesamiento/graficas_sl/correlación.png", width=900)

   with tab2:
      st.header("Resultados - Comprobación de normalidad")
      st.write("Para las variables extraídas se realiza una prueba de normalidad, comprobando que todas las variables del conjunto global de datos no se comportan de manera normal. Debido a que posteriormente el modelo de regresión no obtuvo valor predictivo para el subconjunto de datos de Colombia, no se profundizaron las relaciones específicas en este segmento de datos.")
      ruta_csv = r'Procesamiento/graficas_sl/normal_concatenado.csv'
      # Cargar el DataFrame desde el archivo CSV
      norm_concatenado = pd.read_csv(ruta_csv)
      # Mostrar el DataFrame en Streamlit
      st.dataframe(norm_concatenado)
      

   with tab1:
      st.header("Grafica de Dispersion entre variables")
      st.write ("Para entender la relación entre las variables, se presentan gráficos de dispersión que conectan cada variable independiente con la variable objetivo. Estos gráficos sugieren una tendencia de relación directa entre la corrupción y el desarrollo económico. En resumen, mejores condiciones en la lucha contra la corrupción están asociadas con un mayor desarrollo económico.")
      st.image("Procesamiento/graficas_sl/diagrama_dispersión.png", width=900)

   with tab5:
      st.header("Análisis componentes principales")
      st.write("El hecho de que con 2 o 3 componentes principales se explique más del 95% de la varianza sugiere que estos componentes capturan la mayoría de la información de las variables de corrupción. El gráfico de varianza explicada acumulativa es útil para determinar cuántos componentes son necesarios para conservar una cantidad significativa de varianza.")
      st.write("En el análisis de componentes principales se encontró que los indicadores de corrupción conservan la mayor cantidad de información por medio de 2 componentes principales, el primero con pesos generalmente altos para todas las variables y el segundo tiene un peso significativamente alto para la variable 'CRP_GE.EST' y un peso negativo para 'CRP_VA.EST', lo que sugiere que el segundo componente podría estar relacionado con variaciones específicas en estas dos variables.")
      tab1, tab2 = st.tabs(["📈 Grafica-1", "📈 Grafica-2"])
      with tab1:
         tab1.subheader("Varianza Explicada Acumulativa")
         st.image("Procesamiento/graficas_sl/varianza_explicada_acumulativa.png", width=900)
      with tab2:
         tab2.subheader("Resultados PCA")
         st.image("Procesamiento/graficas_sl/Resultados_pca.png", width=900)
 
   with tab4:
      st.write("Los modelos con mayor capacidad predictiva se identificaron al utilizar el conjunto de datos completo, incorporando la variable país como una categoría esencial. Estos modelos demostraron su eficacia al prever la variable dependiente, especialmente cuando se entrenaron con datos desde 1996 hasta 2002, extendiendo su capacidad predictiva a partir de 1997. Aunque el método de random forest mostró una precisión superior al incluir la variable país, su capacidad predictiva sigue siendo inferior en comparación con los modelos basados en regresión lineal.")
      st.write("La tabla de resultados  presenta un resumen de los modelos desarrollados para este análisis, detallando el Error Cuadrático Medio (MSE) -indicador de un mejor ajuste cuando se acerca a cero- y el R2 -indicador de un mejor ajuste cuando se acerca a 1- para cada modelo. Además, se incluyen gráficos que contrastan la variable dependiente original con las predicciones generadas por los modelos.")
      tab4.subheader("Resultado - Tabla comparativa entre modelos empleados")
      ruta_matriz_modelos = r'Procesamiento/graficas_sl/modelos_df.csv'
      matriz_modelos = pd.read_csv(ruta_matriz_modelos, index_col=0)
      # Aplicar estilos para resaltar valores
      estilos = matriz_modelos.style.background_gradient(cmap='coolwarm').highlight_null('red')
      st.dataframe(matriz_modelos)
      st.header("Grafica Modelos aplicados")
      tab4.subheader ("Gráfica de Resultados")
      tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Grafica-1", "📈 Grafica-2", "📈 Grafica-3","📈 Grafica-4", "📈 Grafica-5", "📈 Grafica-6"])
      with tab1:
         tab1.subheader("1. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO DE REGRESIÓN, SIN EL PAÍS COMO VARIABLE CATEGÓRICA")
         st.image("Procesamiento/graficas_sl/modelo2_rf_sp.png", width=900)
      
      with tab2:
         tab2.subheader("2. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO DE REGRESIÓN, CON EL PAÍS COMO VARIABLE CATEGÓRICA")
         st.image("Procesamiento/graficas_sl/modelo2_rf_cp.png", width=900)
      
      with tab3:
         tab3.subheader("3. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO DE REGRESIÓN, CON EL PAÍS COMO VARIABLE CATEGÓRICA, CON DATOS DE 1996 A 2002")
         st.image("Procesamiento/graficas_sl/modelo2_cp_pred.png", width=900)
      
      with tab4:
         tab4.subheader("4. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO DE REGRESIÓN, CON EL PAÍS COMO VARIABLE CATEGÓRICA, ENTRENADO CON DATOS ANTES DE 2002 Y PROBADO CON DATOS A PARTIR DEL 2003.")
         st.image("Procesamiento/graficas_sl/modelo2_cp_test.png", width=900)

      with tab5:
         tab5.subheader("5. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO RANDOM FOREST, SIN EL PAÍS COMO VARIABLE CATEGÓRICA")
         st.image("Procesamiento/graficas_sl/modelo2_rf_sp.png", width=900)

      with tab6:
         tab6.subheader("6. COMPARACIÓN DE VALORES REALES Y PREDICHOS POR EL MODELO DE REGRESIÓN, CON EL PAÍS COMO VARIABLE CATEGÓRICA")
         st.image("Procesamiento/graficas_sl/modelo2_rf_cp.png", width=900)

 