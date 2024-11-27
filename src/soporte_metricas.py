# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la regresión lineal y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)

from sklearn.model_selection import KFold,LeaveOneOut, cross_val_score


from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


# Ignorar los warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

def generar_df_comparador(X_train,X_test,y_train,y_test,y_train_pred,y_test_pred,nombre_columna_predicciones = "vr_prediccion"):
    """
    Combina características originales, la variable objetivo y las predicciones en un único DataFrame.

    Esta función toma las divisiones de entrenamiento y prueba (X_train, X_test, y_train, y_test) junto con las predicciones 
    del modelo para cada conjunto (y_train_pred, y_test_pred), y genera un DataFrame consolidado que contiene:
    - Las características originales.
    - La variable objetivo real.
    - Las predicciones del modelo.

    Args:
        X_train (pd.DataFrame): DataFrame con las características del conjunto de entrenamiento.
        X_test (pd.DataFrame): DataFrame con las características del conjunto de prueba.
        y_train (pd.DataFrame o pd.Series): Variable objetivo real para el conjunto de entrenamiento.
        y_test (pd.DataFrame o pd.Series): Variable objetivo real para el conjunto de prueba.
        y_train_pred (array-like): Predicciones generadas por el modelo para el conjunto de entrenamiento.
        y_test_pred (array-like): Predicciones generadas por el modelo para el conjunto de prueba.
        nombre_columna_predicciones (str): Nombre que se le asignará a la columna de predicciones en el DataFrame resultante.
            Por defecto, "vr_prediccion".

    Returns:
        pd.DataFrame: Un DataFrame consolidado que incluye:
            - Las características originales.
            - La variable objetivo real.
            - Las predicciones generadas por el modelo.

    Notas:
        - La función asume que las entradas están correctamente alineadas. Es decir, el orden de los datos en las divisiones 
          de entrenamiento y prueba coincide con el de las predicciones.
        - Los índices de los DataFrames de entrada se reordenan para asegurar que los datos originales y las predicciones 
          estén alineados.

    Ejemplo de uso:
        >>> df_comparador = generar_df_comparador(
                X_train=Xtarget_train,
                X_test=Xtarget_test,
                y_train=ytarget_train,
                y_test=ytarget_test,
                y_train_pred=ytarget_train_pred,
                y_test_pred=ytarget_test_pred,
                nombre_columna_predicciones="price_pred"
            )
    """
    # Generar df Prediccion Train
    df_y_train = pd.DataFrame(y_train_pred)
    df_y_train = df_y_train.rename(columns={0:nombre_columna_predicciones})
    # Generar df Prediccion Test
    df_y_test = pd.DataFrame(y_test_pred)
    df_y_test = df_y_test.rename(columns={0:nombre_columna_predicciones})
    # Concatenarlos
    predicciones = pd.concat([df_y_train,df_y_test],ignore_index=True)

    # Reseteamos el índice sin eliminar el original (para ordenarlo tal y como estaba!)
    X_train_reset = X_train.reset_index(drop=False)
    X_test_reset = X_test.reset_index(drop=False)
    y_train_reset = y_train.reset_index(drop=False)
    y_test_reset = y_test.reset_index(drop=False)
    # Concatenar los DataFrames
    X_desordenado = pd.concat([X_train_reset, X_test_reset])
    y_desordenado = pd.concat([y_train_reset, y_test_reset])

    # Concatenamos la "y" con "predicciones"
    # De forma que tendremos La Variable Respuesta Original con las predicciones
    # Y de está forma me aseguro de que están los precios alineados
    precios = pd.concat([y_desordenado.reset_index(drop=True), predicciones.reset_index(drop=True)], axis=1)

    # Ordenamos "X" y "precios" por la columna index
    X_desordenado = X_desordenado.sort_values(by=("index"))
    precios = precios.sort_values(by=("index"))

    #Nos quitamos la columna index y la reseteamos
    X_desordenado.drop(columns="index",inplace=True)
    precios.drop(columns="index",inplace=True)
    X_desordenado.reset_index(drop=True,inplace=True)
    precios.reset_index(drop=True,inplace=True)

    # Concatenamos ambos para obtener el DataFrame original con las predicciones
    df_comparador = pd.concat([X_desordenado, precios], axis=1)
    return df_comparador


def obtener_metricas(y_train,y_pred_train,y_test,y_pred_test):
    """
    Calcula y devuelve métricas de evaluación para un modelo de regresión en conjuntos de entrenamiento y prueba.

    Parámetros:
    -----------
    y_train : array-like
        Valores reales del conjunto de entrenamiento.
    y_pred_train : array-like
        Predicciones del modelo para el conjunto de entrenamiento.
    y_test : array-like
        Valores reales del conjunto de prueba.
    y_pred_test : array-like
        Predicciones del modelo para el conjunto de prueba.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las métricas de evaluación para los conjuntos de entrenamiento y prueba. 
        Las métricas incluyen:
            - r2_score: Coeficiente de determinación.
            - MAE: Error absoluto medio.
            - MSE: Error cuadrático medio.
            - RMSE: Raíz cuadrada del error cuadrático medio.

    Ejemplo:
    --------
    metricas = obtener_metricas(
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test
    )
    print(metricas)
    """

    metricas = {
        'train': {
        'r2_score': r2_score(y_train, y_pred_train),
        'MAE': mean_absolute_error(y_train, y_pred_train),
        'MSE': mean_squared_error(y_train, y_pred_train),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train))
        },
        'test': {
        'r2_score': r2_score(y_test, y_pred_test),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
    }
    df_metricas = pd.DataFrame(metricas).T
    return df_metricas

def obtener_metricas_logistica(y_train, y_pred_train, y_test, y_pred_test, prob_train, prob_test):
   
    metricas = {
        'train': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted'),
            'kappa': cohen_kappa_score(y_train,y_pred_train),
            'auc': roc_auc_score(y_train,prob_train)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted'),
            'kappa': cohen_kappa_score(y_test,y_pred_test),
            'auc': roc_auc_score(y_test,prob_test)
        }
    }
    df_metricas = pd.DataFrame(metricas)
    return df_metricas

def comparar_arboles(df_previos, df_final,lista_previos=False,nombre_modelo = "modelo_previo"):
    """
    Compara dos DataFrames de métricas (inicial y final) y calcula las diferencias porcentuales
    entre ellos para mostrar mejoras o empeoramientos en las métricas.

    Args:
        df_previo (pd.DataFrame): DataFrame con las métricas iniciales.
        df_final (pd.DataFrame): DataFrame con las métricas finales.
    """
    if lista_previos == True:
        df_unir = pd.DataFrame()
        
        for i, df in enumerate(df_previos):
            df.reset_index(inplace=True)
            df = df.rename(columns={"index":"entrenamiento"})
            df["modelo"] = f"modelo {i}"
            df = df[["modelo","entrenamiento","r2_score","MAE","MSE","RMSE"]]
            
            df_unir = pd.concat([df_unir,df], axis=0)
        
        df_final.reset_index(inplace=True)
        df_final = df_final.rename(columns={"index":"entrenamiento"})
        df_final["modelo"] = f"modelo final"
        df_final = df_final[["modelo","entrenamiento","r2_score","MAE","MSE","RMSE"]]
        df_unido = pd.concat([df_unir,df_final])
        display(df_unido)
    else:
        df_previos.reset_index(inplace=True)
        df_previos = df_previos.rename(columns={"index":"entrenamiento"})
        df_previos["modelo"] = f"{nombre_modelo}"
        df_previos = df_previos[["modelo","entrenamiento","r2_score","MAE","MSE","RMSE"]]
        
        df_final.reset_index(inplace=True)
        df_final = df_final.rename(columns={"index":"entrenamiento"})
        df_final["modelo"] = f"modelo final"
        df_final = df_final[["modelo","entrenamiento","r2_score","MAE","MSE","RMSE"]]
        df_unido = pd.concat([df_previos,df_final])
        display(df_unido)

def residual_plot(df_comparador,nombre_col_original,nombre_col_prediccion):
    """
    Genera un residual plot comparando los valores originales y las predicciones de un modelo.

    Args:
        df_comparador (pd.DataFrame): DataFrame que contiene las columnas con los valores originales y las predicciones.
        nombre_col_original (str): Nombre de la columna que contiene los valores originales (reales).
        nombre_col_prediccion (str): Nombre de la columna que contiene los valores predichos por el modelo.

    Returns:
        None: La función genera un gráfico de residuos y no devuelve ningún valor.
    
    Side Effects:
        - Modifica el DataFrame de entrada añadiendo una columna llamada 'residual' con los valores de los residuos.
        - Muestra un gráfico de residuos (residual plot) usando Matplotlib y Seaborn.

    Raises:
        ValueError: Si las columnas especificadas no existen en el DataFrame.
    
    Notas:
        - Los residuos se calculan como la diferencia entre los valores originales y las predicciones: 
          residuo = valor_original - predicción.
        - El gráfico incluye una línea de referencia en el eje 0 para visualizar mejor las desviaciones.

    Ejemplo de uso:
        >>> residual_plot(df_comparador, 'valor_real', 'predicciones_modelo')
    """
    # Calcular los residuales
    df_comparador['residual'] = df_comparador[nombre_col_original] - df_comparador[nombre_col_prediccion]

    # Crear el residual plot
    plt.figure(figsize=(12, 6))
    sns.residplot(x=nombre_col_original, y="residual", data=df_comparador, lowess=True, color="purple")

    # Personalizar el gráfico
    plt.axhline(0, color='red', linestyle='--', label='Residual = 0')
    plt.title("Residual Plot: Valor Original vs. Predicciones")
    plt.xlabel("Valor Original")
    plt.ylabel("Residuales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def scatterplot_regresion(df_comparador,nombre_col_original,nombre_col_prediccion):
    """
    Genera un scatter plot con línea de regresión para comparar valores originales y predicciones.

    Args:
        df_comparador (pd.DataFrame): DataFrame que contiene las columnas de valores originales y predicciones.
        nombre_col_original (str): Nombre de la columna que contiene los valores originales (reales).
        nombre_col_prediccion (str): Nombre de la columna que contiene los valores predichos por el modelo.

    Returns:
        None: La función genera un gráfico y no devuelve ningún valor.

    Notas:
        - Utiliza `sns.lmplot` para generar el gráfico con una línea de regresión ajustada a los datos.
        - La línea de regresión está personalizada con color rojo (`line_kws={'color': 'red'}`).
        - La relación entre valores originales y predicciones debería estar cerca de una línea diagonal si el modelo
          tiene un buen desempeño.

    Ejemplo de uso:
        >>> scatterplot_regresion(df_comparador, 'valor_real', 'valor_predicho')
    """
    # Crear el scatter plot con línea de regresión
    sns.lmplot(x=nombre_col_original, y=nombre_col_prediccion, data=df_comparador, line_kws={'color': 'red'})

    # Personalizar el gráfico
    plt.title(" Original vs. Predicción")
    plt.tight_layout()
    plt.show()
