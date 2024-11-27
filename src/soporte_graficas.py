import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns 

def visualizar_categoricas(dataframe,lista_cols_categoricas,variable_respuesta,tipo_grafica = "boxplot", grafica_size=(15,10),paleta="mako",barplot_calc="mean"):
    """
    Visualiza la relación entre variables categóricas y una variable de respuesta utilizando boxplots o barplots.

    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame que contiene los datos para la visualización.
    lista_cols_categoricas : list
        Lista de nombres de columnas categóricas en el DataFrame que se quieren analizar.
    variable_respuesta : str
        Nombre de la variable de respuesta (columna dependiente) que se analizará en relación con las categóricas.
    tipo_grafica : str, opcional
        Tipo de gráfica a generar. Puede ser "boxplot" o "barplot". Por defecto es "boxplot".
    grafica_size : tuple, opcional
        Tamaño de las gráficas en formato (ancho, alto). Por defecto es (15, 10).
    paleta : str, opcional
        Paleta de colores para las gráficas. Por defecto es "mako".
    barplot_calc : str o callable, opcional
        Métrica para calcular los valores en el barplot. Por defecto es "mean".

    Errores:
    --------
    ValueError
        Si el valor de `tipo_grafica` no es "boxplot" o "barplot".

    Notas:
    ------
    - Si hay un número impar de columnas categóricas, se elimina el último eje vacío de las subgráficas.
    - En el caso de los boxplots, se utiliza `whis=1.5` para definir el alcance de los bigotes.

    Ejemplo:
    --------
    visualizar_categoricas(
        dataframe=df,
        lista_cols_categoricas=['col1', 'col2'],
        variable_respuesta='target',
        tipo_grafica='barplot',
        grafica_size=(10, 8),
        paleta='viridis',
        barplot_calc='median'
    )
    """

    num_filas = math.ceil(len(lista_cols_categoricas)/2)

    fig , axes = plt.subplots(ncols=2 , nrows=num_filas,figsize= grafica_size)
    axes = axes.flat

    for indice, columna in enumerate(lista_cols_categoricas):
        if tipo_grafica.lower() == "boxplot":
            sns.boxplot(x= columna,
                        y= variable_respuesta,
                        data=dataframe,
                        whis = 1.5,
                        hue=columna,
                        legend=False,
                        ax = axes[indice])
        elif tipo_grafica.lower() == "barplot":
            sns.barplot(x=columna,
                        y= variable_respuesta,
                        ax = axes[indice],
                        estimator=barplot_calc,
                        palette=paleta,
                        data=dataframe)
        else:
            print("Debes elegir entre boxplot y barplot")
    
        axes[indice].set_title(f"Relación {columna} con {variable_respuesta}")
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation=90)

    if num_filas % 2 != 0:
        fig.delaxes(axes[-1])   
    plt.tight_layout()


def boxplot_scaler(df,columnas_plotear,scaler="Scaler",grafica_size = (15, 10)):
    """
    Genera boxplots para un conjunto de columnas de un DataFrame, mostrando la distribución de los valores tras aplicar un escalado.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene los datos a visualizar.
    columnas_plotear : list
        Lista de nombres de las columnas para las que se generarán los boxplots.
    scaler : str, opcional
        Nombre o descripción del escalador aplicado a las columnas, para mostrarlo en los títulos de las gráficas. Por defecto es "Scaler".
    grafica_size : tuple, opcional
        Tamaño de las gráficas en formato (ancho, alto). Por defecto es (15, 10).

    Notas:
    ------
    - Si hay un número impar de columnas en `columnas_plotear`, se elimina el último eje vacío de las subgráficas.
    - El título de cada boxplot incluye el nombre del escalador especificado y el nombre de la columna.

    Ejemplo:
    --------
    boxplot_scaler(
        df=datos_escalados,
        columnas_plotear=['col1', 'col2', 'col3'],
        scaler='StandardScaler',
        grafica_size=(12, 8)
    )
    """

    num_filas = math.ceil(len(columnas_plotear)/2)
    fig , axes = plt.subplots(nrows=num_filas , ncols=2, figsize = grafica_size)
    axes = axes.flat
    for indice, columna in enumerate(columnas_plotear):
        sns.boxplot(x = columna, data = df, ax = axes[indice])
        axes[indice].set_title(f"{scaler} {columna}")
    
    if num_filas % 2 != 0:
        fig.delaxes(axes[-1])   

    plt.tight_layout()