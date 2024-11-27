# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

def exploracion_dataframe(dataframe, columna_control, estadisticos = False):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    # Tipos de columnas
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    # Enseñar solo las columnas categoricas (o tipo objeto)
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    if estadisticos == True:
        for categoria in dataframe[columna_control].unique():
            dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
            #Describe de objetos
            print("\n ..................... \n")

            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)

            #Hacer un describe
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
    else: 
        pass

def separarar_df(dataframe):
    """
    Separa un DataFrame en dos subconjuntos: uno con columnas numéricas y otro con columnas categóricas.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame a separar.

    Retorna:
    --------
    tuple
        Una tupla con dos elementos:
            - El primer elemento es un DataFrame que contiene únicamente las columnas numéricas.
            - El segundo elemento es un DataFrame que contiene únicamente las columnas categóricas (tipo objeto).

    Ejemplo:
    --------
    df_numerico, df_categorico = separarar_df(dataframe)

    Notas:
    ------
    - Las columnas numéricas se identifican con `select_dtypes(include=np.number)`.
    - Las columnas categóricas se identifican con `select_dtypes(include="O")`.
    """

    return dataframe.select_dtypes(include=np.number), dataframe.select_dtypes(include="O")

def plot_numericas(dataframe,grafica_size = (15,10)):
    """
    Genera histogramas para visualizar la distribución de las variables numéricas en un DataFrame.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las columnas numéricas a graficar.
    grafica_size : tuple, opcional
        Tamaño de la figura para los gráficos generados, en formato (ancho, alto). Por defecto es (15, 10).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra una figura con histogramas para cada columna numérica del DataFrame.

    Notas:
    ------
    - Las columnas a graficar se toman directamente del DataFrame proporcionado.
    - Si el número de columnas numéricas es impar, se elimina el último eje vacío para evitar espacios sin uso.
    - Cada gráfico incluye un título con el nombre de la columna correspondiente.

    Ejemplo:
    --------
    plot_numericas(
        dataframe=df_numerico,
        grafica_size=(12, 8)
    )
    """

    cols_numericas = dataframe.columns
    filas = math.ceil(len(cols_numericas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for i, col in enumerate(cols_numericas):
        sns.histplot(x= col,data=dataframe,ax= axes[i])
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    plt.tight_layout()

def plot_categoricas(dataframe, paleta="mako",grafica_size = (15,10)):
    """
    Genera gráficos de barras para visualizar la distribución de las variables categóricas en un DataFrame.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las columnas categóricas a graficar.
    paleta : str, opcional
        Paleta de colores para los gráficos. Por defecto es "mako".
    grafica_size : tuple, opcional
        Tamaño de la figura para los gráficos generados, en formato (ancho, alto). Por defecto es (15, 10).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra una figura con gráficos de barras para cada columna categórica del DataFrame.

    Notas:
    ------
    - Las columnas a graficar se toman directamente del DataFrame proporcionado.
    - Las barras se ordenan por frecuencia descendente.
    - Si el número de columnas categóricas es impar, se elimina el último eje vacío para evitar espacios sin uso.
    - Cada gráfico incluye un título con el nombre de la columna correspondiente, y las etiquetas del eje X se rotan para mejorar la legibilidad.

    Ejemplo:
    --------
    plot_categoricas(
        dataframe=df_categorico,
        paleta="viridis",
        grafica_size=(12, 8)
    )
    """

    cols_categoricas = dataframe.columns
    filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for i, col in enumerate(cols_categoricas):
        sns.countplot(  x= col,data=dataframe,
                        ax= axes[i],
                        hue = col,
                        palette=paleta,
                        order=dataframe[col].value_counts().index,
                        legend=False)
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].tick_params(rotation=90)
    
    plt.tight_layout()
    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    

def relacion_vr_categoricas(dataframe,variable_respuesta,paleta="mako",grafica_size = (15,10)):
    """
    Visualiza la relación entre una variable de respuesta y las variables categóricas de un DataFrame.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las variables categóricas y la variable de respuesta.
    variable_respuesta : str
        Nombre de la variable de respuesta, cuyo promedio se calculará para cada categoría.
    paleta : str, opcional
        Paleta de colores para los gráficos. Por defecto es "mako".
    grafica_size : tuple, opcional
        Tamaño de la figura para los gráficos generados, en formato (ancho, alto). Por defecto es (15, 10).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra gráficos de barras para cada columna categórica, mostrando la media de la variable de respuesta.

    Notas:
    ------
    - Las columnas categóricas se obtienen utilizando la función `separarar_df`.
    - Los valores de la variable de respuesta se agrupan por categoría y se calcula su promedio.
    - Los gráficos están ordenados por el valor promedio descendente de la variable de respuesta.
    - Si el número de columnas categóricas es impar, se elimina el último eje vacío para evitar espacios sin uso.

    Ejemplo:
    --------
    relacion_vr_categoricas(
        dataframe=datos,
        variable_respuesta="precio",
        paleta="viridis",
        grafica_size=(12, 8)
    )
    """

    df_cat = separarar_df(dataframe)[1]
    cols_categoricas = df_cat.columns
    filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for indice,columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta,ascending=False)
        sns.barplot(x= columna,
                    y= variable_respuesta,
                    data = datos_agrupados,
                    ax= axes[indice],
                    hue= columna,
                    legend=False,
                    palette=paleta)
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre: {columna} y {variable_respuesta}")
    
    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    
    plt.tight_layout()

def relacion_vr_numericas(dataframe,variable_respuesta,paleta="mako",grafica_size = (15,10)):
    """
    Visualiza la relación entre una variable de respuesta y las variables numéricas de un DataFrame.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las variables numéricas y la variable de respuesta.
    variable_respuesta : str
        Nombre de la variable de respuesta que se analizará en relación con las demás variables numéricas.
    paleta : str, opcional
        Paleta de colores para los gráficos. Por defecto es "mako".
    grafica_size : tuple, opcional
        Tamaño de la figura para los gráficos generados, en formato (ancho, alto). Por defecto es (15, 10).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra un conjunto de diagramas de dispersión para cada columna numérica en relación con la variable de respuesta.

    Notas:
    ------
    - Las columnas numéricas se obtienen utilizando la función `separarar_df`.
    - Se omite la variable de respuesta (`variable_respuesta`) en los diagramas de dispersión.
    - Cada gráfico incluye una relación bivariada entre una columna numérica y la variable de respuesta.
    - Si el número de columnas numéricas es impar, se elimina el último eje vacío para evitar espacios sin uso.

    Ejemplo:
    --------
    relacion_vr_numericas(
        dataframe=datos,
        variable_respuesta="precio",
        paleta="viridis",
        grafica_size=(12, 8)
    )
    """

    numericas = separarar_df(dataframe)[0]
    cols_numericas = numericas.columns
    filas = math.ceil(len(cols_numericas)/2)
    fig, axes = plt.subplots(nrows= filas,ncols=2,figsize = grafica_size)
    axes = axes.flat

    for indice,columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x = columna,
                            y = variable_respuesta,
                            data = numericas,
                            ax = axes[indice],
                            hue = columna,
                            legend = False,
                            palette=paleta)
            axes[indice].set_title(columna)
    
    plt.tight_layout()

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    
def matriz_correlacion(dataframe, grafica_size=(10,7)):
    """
    Genera un mapa de calor para visualizar la matriz de correlación de las variables numéricas de un DataFrame.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las columnas numéricas para calcular la matriz de correlación.
    grafica_size : tuple, opcional
        Tamaño de la figura del mapa de calor, en formato (ancho, alto). Por defecto es (10, 7).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra un mapa de calor con la matriz de correlación de las variables numéricas.

    Notas:
    ------
    - La correlación se calcula únicamente para las columnas numéricas.
    - Se aplica una máscara triangular superior para ocultar valores redundantes en el mapa de calor.
    - Los valores de correlación están anotados en el mapa para facilitar su interpretación.

    Ejemplo:
    --------
    matriz_correlacion(
        dataframe=datos,
        grafica_size=(12, 8)
    )
    """

    plt.figure(figsize=grafica_size)
    matriz_corr = dataframe.corr(numeric_only=True)
    mascara = np.triu(np.ones_like(matriz_corr,dtype = np.bool_))
    sns.heatmap(matriz_corr,
                annot=True,
                vmin= -1,
                vmax=1,
                mask = mascara)
    
def detectar_outliers(dataframe,colorear="orange",grafica_size = (15,10)):
    """
    Detecta y visualiza valores atípicos (outliers) en las columnas numéricas de un DataFrame utilizando boxplots.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene las variables numéricas a analizar.
    colorear : str, opcional
        Color de las cajas del boxplot. Por defecto es "orange".
    grafica_size : tuple, opcional
        Tamaño de la figura para los gráficos generados, en formato (ancho, alto). Por defecto es (15, 10).

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra una figura con boxplots para cada columna numérica del DataFrame.

    Notas:
    ------
    - Las columnas numéricas se obtienen utilizando la función `separarar_df`.
    - Los valores atípicos se muestran en rojo para facilitar su identificación.
    - Si el número de columnas numéricas es impar, se elimina el último eje vacío para evitar espacios sin uso.
    - Cada gráfico incluye un título que indica el nombre de la columna correspondiente.

    Ejemplo:
    --------
    detectar_outliers(
        dataframe=datos,
        colorear="blue",
        grafica_size=(12, 8)
    )
    """

    df_num = separarar_df(dataframe)[0]
    num_filas =  math.ceil(len(df_num.columns)/2)

    fig, axes = plt.subplots(ncols=2,nrows=num_filas, figsize= grafica_size)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):

        sns.boxplot(x=columna,
                    data = df_num,
                    ax= axes[indice],
                    color= colorear,
                    flierprops = {"markersize":5,"markerfacecolor":"red"})
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")
    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()

def diferencia_tras_rellenar_nulos(df_before, df_after):
    """
    Compara las estadísticas descriptivas de un DataFrame antes y después de rellenar valores nulos, mostrando los cambios en porcentaje.

    Parámetros:
    -----------
    df_before : pd.DataFrame
        DataFrame original, antes de realizar la operación de relleno de nulos.
    df_after : pd.DataFrame
        DataFrame modificado, después de realizar la operación de relleno de nulos.

    Retorna:
    --------
    None
        No retorna ningún valor, pero muestra:
        - Estadísticas descriptivas antes de la operación (`df_before`).
        - Estadísticas descriptivas después de la operación (`df_after`).
        - Porcentaje de cambio entre ambos DataFrames.

    Notas:
    ------
    - Las estadísticas incluyen métricas como media, mediana y desviación estándar, extraídas con `describe()`.
    - Los cambios se calculan como porcentaje relativo entre `df_after` y `df_before`.
    - Los valores NaN en la diferencia porcentual se rellenan con 0 para evitar inconsistencias en los resultados.

    Ejemplo:
    --------
    diferencia_tras_rellenar_nulos(
        df_before=datos_antes,
        df_after=datos_despues
    )
    """

    # Obtener el describe de ambos DataFrames y solo media mediana y desviacion estándar

    describe_before = df_before.describe().T
    describe_after = df_after.describe().T
        
    # Calcular el porcentaje de cambio
    difference_percent = ((describe_after - describe_before) / describe_before) * 100
        
    # Mostrar las tres tablas: describe antes, describe después y porcentaje de cambio
    print("\n ..................... \n")
    print("Estadísticas antes de la operación:")
    display(describe_before)
    print("\n ..................... \n")
    print("Estadísticas después de la operación:")
    display(describe_after)
    print("\n ..................... \n")
    print("Diferencia porcentual:")
    display(difference_percent.fillna(0))  # Llenar NaN con 0 en caso de que no haya cambios
