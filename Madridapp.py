#Importar librerias 
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np 
from funpymodeling.exploratory import freq_tbl 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

@st.cache_resource

#Crear la función de carga de datos
def load_data():
    #lectura del archivo sin indice
    df1=pd.read_csv("df1.csv")
    #lectura del archivo con indice
    df2=pd.read_csv("df1.csv", index_col="name")

        #Etapa de procesamiento de datos
    #ANALISIS UNIVARIADO DE FRECUENCIAS

    #obtener un analisi univariado de una variable categorica en especifico
    table=freq_tbl(df1["host_is_superhost"])
    #obtener un filtro de los valores mas relevantes de la variable categorica seleccionada
    Filtro=table[table["frequency"]>1]
    #ajusto el indice de mi dataframe
    Filtro_index1=Filtro.set_index("host_is_superhost")
    ###############
    table2=freq_tbl(df1["room_type"])
    #obtener un filtro de los valores mas relevantes de la variable categorica seleccionada
    Filtro2=table2[table2["frequency"]>1]
    #ajusto el indice de mi dataframe
    Filtro_index2=Filtro2.set_index("room_type")
    ###############
    table3=freq_tbl(df1["neighbourhood_group_cleansed"])
    #obtener un filtro de los valores mas relevantes de la variable categorica seleccionada
    Filtro3=table3[table3["frequency"]>1]
    #ajusto el indice de mi dataframe
    Filtro_index3=Filtro3.set_index("neighbourhood_group_cleansed")
    #Seleccionar las columnas de tipo numericas del dataframe Filtro_index1
    numeric_df1=Filtro_index1.select_dtypes(["float","int"])
    numeric_cols1=numeric_df1.columns

    #Seleccionar las columnas de tipo numericas del dataframe 2
    numeric_df2=df2.select_dtypes(["float","int"])
    numeric_cols2=numeric_df2.columns

    binary_cols=df1[["host_has_profile_pic", "host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability", "instant_bookable"]]

    return Filtro_index1, Filtro_index2, Filtro_index3, df2, numeric_df1, numeric_cols1, numeric_df2, numeric_cols2, binary_cols

Filtro_index1, Filtro_index2, Filtro_index3, df2, numeric_df1, numeric_cols1, numeric_df2, numeric_cols2, binary_cols =load_data()


#1 Creación de la SIDEBAR
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/vector-gratis/fondo-abstracto-blanco_23-2148810113.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("DASHBOARD MADRID")
st.sidebar.image("logo.png", width=150)
st.sidebar.markdown("---")
st.sidebar.subheader("Panel de seleccion")
Frames=st.sidebar.selectbox(label="Frames", options=["Análisis Univariado", "Barplot", "Regresión Lineal", "Regresión Multiple", 
"Regresión No Lineal", "Regresión Logística"])

if Frames=="Análisis Univariado":
    #Generar encabezados para el dashboard
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Madrid")
    st.image("que-visitar-en-madrid.jpg")
    st.write("Este dashboard muestra un análisis univariado de las variables categóricas más importantes de los datos de Airbnb en la ciudad de Madrid.")
    st.subheader("Tipos de cuarto:")
    st.write(Filtro_index2)

    st.subheader("Superhost:")
    st.write(Filtro_index1)

    st.subheader("Vecindario:")
    st.write(Filtro_index3)

    check_box=st.checkbox(label="Mostrar Dataset")
    #Condicional para que aparezca el checkbox
    if check_box:
        st.write(df2)

    Vars_num_selected=st.sidebar.multiselect(label="Variables graficadas", options=numeric_cols2)
    lineplot1 = px.line(data_frame=df2, x=df2.index, 
                  y= Vars_num_selected, title= str('Caracteristicas por nombre de alojamiento'), 
                  width=1600, height=600)
    #Mostramos el lineplot
    checkbox2=st.checkbox(label="Mostrar Lineplot")  
    if checkbox2:
        st.plotly_chart(lineplot1)


##########
if Frames == "Barplot":
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Barplot Análisis Univariado")
   
    # Selección de la variable categórica para graficar
    Variables = st.selectbox(label= "Variable categórica", 
                                     options= ["Tipo de cuarto", 
                                               "Superhost", 
                                               "Vecindario"])
    
    # Selección de la métrica numérica para graficar
    Vars_Num = st.selectbox(label= "Tipo de dato", 
                            options= ["frequency", "percentage", "cumulative_perc"])

    color1 = ["#e826ff"]
    color2 = ["#ff3b82"]
    color3 = ["#8273ff"]
    
    # Dependiendo de la selección en el selectbox de Variables, mostramos una gráfica
    if Variables == "Tipo de cuarto":
        figure = px.bar(data_frame=Filtro_index2, 
                        x=Filtro_index2.index, 
                        y=Vars_Num,
                        color_discrete_sequence=color1)
    
    elif Variables == "Superhost":
        figure = px.bar(data_frame=Filtro_index1, 
                        x=Filtro_index1.index, 
                        y=Vars_Num,
                        color_discrete_sequence=color2)
    
    elif Variables == "Vecindario":
        figure = px.bar(data_frame=Filtro_index3, 
                        x=Filtro_index3.index, 
                        y=Vars_Num,
                        color_discrete_sequence=color3)
     
    # Mostramos la gráfica seleccionada
    st.plotly_chart(figure)

if Frames=="Regresión Lineal":
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Regresión lineal")

    color1 = ["#2becec"]

    #Generar dos cuadros de multiseleccion (Y) para seleccionar variables a graficar
    x_selected=st.sidebar.selectbox(label="Seleccione la variable x", options=numeric_cols2)
    y_selected=st.sidebar.selectbox(label="Seleccione la variable y", options=numeric_cols2)
    
    correlacion = np.corrcoef(numeric_df2[x_selected], numeric_df2[y_selected])[0, 1]

    figure3=px.scatter(data_frame=numeric_df2, x=x_selected, y=y_selected,color_discrete_sequence=color1,
    title=f"Dispersiones (Coeficiente de correlación: {correlacion:.2f})")
    st.plotly_chart(figure3)

        # Calcular la matriz de correlación de las variables numéricas
    correlacion_matrix = numeric_df2.corr()

    # Crear el heatmap usando plotly
    figure_heatmap = px.imshow(correlacion_matrix, 
                               text_auto=True, # Mostrar los valores en cada celda
                               aspect="auto",  # Ajustar la proporción
                               color_continuous_scale="Blues",  # Escala de colores
                               title="Mapa de calor de correlaciones")
    
    Button=st.button(label="Mostrar heatmap")
    if Button:
        st.plotly_chart(figure_heatmap)

if Frames=="Regresión Multiple":
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Regresión Multiple")

    x_selected=st.sidebar.multiselect(label="Seleccione las variables x", options=numeric_cols2)
    y_selected=st.sidebar.selectbox(label="Seleccione la variable y", options=numeric_cols2)
    
    if len(x_selected) == 0 or y_selected == "":
            st.write("Por favor, seleccione al menos una variable independiente (x) y una variable dependiente (y).")
    else:
        # Definir las variables independientes (predictoras) y la variable dependiente
        X = numeric_df2[x_selected]  # Variables predictoras
        y = numeric_df2[y_selected]  # Variable objetivo

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear el modelo de regresión lineal
        modelo = LinearRegression()

        # Ajustar el modelo con los datos de entrenamiento
        modelo.fit(X_train, y_train)

        # Hacer predicciones con los datos de prueba
        y_pred = modelo.predict(X_test)

        # Mostrar el coeficiente de determinación (R2) y el error cuadrático medio
        st.write(f"R2: {r2_score(y_test, y_pred)}")

        # Mostrar los coeficientes del modelo
        st.write("Coeficientes:", modelo.coef_)
        st.write("Intercepto:", modelo.intercept_)

        # Crear un DataFrame para organizar los valores reales y predichos
        resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

        # Graficar los valores reales contra los predichos
        fig = px.scatter(resultados, x='Real', y='Predicción', 
                         title='Valores Reales vs Predicciones',
                         labels={'Real': 'Valores Reales', 'Predicción': 'Valores Predichos'})

        # Añadir una línea de referencia (y=x) para ver qué tan cerca están las predicciones de los valores reales
        fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                      line=dict(color="Red",), xref='x', yref='y')

        # Mostrar la gráfica
        st.plotly_chart(fig)


if Frames == "Regresión No Lineal":
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Regresión No Lineal")

    # Seleccionar variables
    x_selected = st.sidebar.selectbox(label="x", options=numeric_cols2)
    y_selected = st.sidebar.selectbox(label="y", options=numeric_cols2)
    
    # Convertir las columnas seleccionadas en arrays numéricos
    x_data = numeric_df2[x_selected].values
    y_data = numeric_df2[y_selected].values

    # Seleccionar modelo no lineal
    Modelo = st.sidebar.selectbox(label="Modelo Seleccionado", 
                                  options=["Función lineal con producto de coeficientes","Función polinomial inversa"])

        # Modelo Función lineal con producto de coeficientes
    if Modelo == "Función lineal con producto de coeficientes":
        st.write("Función lineal con producto de coeficientes")
        def func2(x, a, b, c):
            return a * x + b * x + c * x

        # Ajustar los parámetros
        parametros, _ = curve_fit(func2, x_data, y_data)
        a, b, c = parametros
        yfit2 = func2(x_data, a, b, c)

        st.write("Coeficiente de determinación:")
        R2 = r2_score(y_data, yfit2)
        st.write(R2) 

        # Graficar los datos originales y ajustados
        nolineal2, ax = plt.subplots()
        ax.plot(x_data, y_data, 'bo', label="y-original")
        ax.plot(x_data, yfit2, 'r-', label=f"y = ({a:.2f} + {b:.2f} + {c:.2f})*x")
        ax.set_xlabel(x_selected)
        ax.set_ylabel(y_selected)
        ax.legend(loc='best', fancybox=True, shadow=True)
        ax.grid(True)

        # Mostrar la gráfica en Streamlit
        st.pyplot(nolineal2)
        
    elif Modelo == "Función polinomial inversa":
        st.write("Función polinomial inversa")
        def func1(x, a, b, c):
            return a * x**2 / b + c * x

        # Ajustar los parámetros de la función curve_fit
        parametros, covs = curve_fit(func1, x_data, y_data)
        a, b, c = parametros[0], parametros[1], parametros[2]
        yfit1 = func1(x_data, a, b, c)

        st.write("Coeficiente de determinación:")
        R2 = r2_score(y_data, yfit1)
        st.write(R2) 

        # Graficar los datos originales y los ajustados
        nolineal1, ax = plt.subplots()
        ax.plot(x_data, y_data, 'bo', label="y-original")
        ax.plot(x_data, yfit1, 'r-', label=f"y = {a:.2f}*x^2/{b:.2f} + {c:.2f}*x")
        ax.set_xlabel(x_selected)
        ax.set_ylabel(y_selected)
        ax.legend(loc='best', fancybox=True, shadow=True)
        ax.grid(True)

        # Mostrar gráfica en Streamlit
        st.pyplot(nolineal1)


if Frames == "Regresión Logística":
    st.title("Inside Airbnb")
    st.markdown("---")
    st.header("Regresión Logística")

    # Selección de variables independientes y dependiente
    Vars_Indep = st.sidebar.multiselect(label="x", options=numeric_cols2)
    Var_Dep = st.sidebar.selectbox(label="y", options=binary_cols.columns)

    if len(Vars_Indep) > 0 and Var_Dep:
        # Variables independientes y dependiente
        X = df2[Vars_Indep]
        y = df2[Var_Dep]

        # Eliminar filas con valores faltantes para evitar problemas en el modelo
        df_model = pd.concat([X, y], axis=1).dropna()
        X = df_model[Vars_Indep].values
        y = df_model[Var_Dep].values

        # Verificar si las dimensiones son consistentes
        if len(X) > 0 and len(y) > 0 and X.shape[0] == y.shape[0]:
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

            # Escalar solo las variables independientes
            escalar = StandardScaler()
            X_train = escalar.fit_transform(X_train)
            X_test = escalar.transform(X_test)

            # Definir y entrenar el modelo de regresión logística
            algoritmo = LogisticRegression()
            algoritmo.fit(X_train, y_train)

            # Realizar predicciones
            y_pred = algoritmo.predict(X_test)

            # Matriz de confusión
            matriz = confusion_matrix(y_test, y_pred)
            st.write('Matriz de Confusión:')
            st.write(matriz)

            # Calcular la precisión del modelo
            precision = precision_score(y_test, y_pred, pos_label='yes')
            st.write('Precisión del modelo label=Yes:')
            st.write(precision)

             # Calcular la sensibilidad del modelo (recall)
            sensibilidad = recall_score(y_test, y_pred, pos_label='yes')
            st.write('Sensibilidad del modelo label=Yes:')
            st.write(sensibilidad)

            # Calcular la exactitud del modelo
            exactitud = accuracy_score(y_test, y_pred)
            st.write('Exactitud del modelo:')
            st.write(exactitud)

            # Calcular la precisión del modelo
            precision2 = precision_score(y_test, y_pred, pos_label='no')
            st.write('Precisión del modelo label=No:')
            st.write(precision2)

            # Calcular la sensibilidad del modelo (recall)
            sensibilidad2 = recall_score(y_test, y_pred, pos_label='no')
            st.write('Sensibilidad del modelo label=No:')
            st.write(sensibilidad2)


            # Valores de la matriz de confusión
            TN, FP, FN, TP = matriz.ravel()

            # Datos para la gráfica de pastel
            labels = ['Verdaderos Negativos', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
            sizes = [TN, FP, FN, TP]
            colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
            explode = (0.1, 0.1, 0.1, 0.1)  # Destacar todas las porciones

            # Crear la gráfica de pastel
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
            ax.axis('equal')  # Para asegurar que el pastel sea un círculo

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

        else:
            st.write("Las variables seleccionadas no tienen el mismo número de filas. Revisa los datos.")
    else:
        st.write("Por favor, selecciona al menos una variable independiente y una variable dependiente.")
