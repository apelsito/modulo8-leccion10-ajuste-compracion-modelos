# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
import xgboost as xgb
import pickle

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Inicialización de probabilidades para ROC
        self.prob_train = None
        self.prob_test = None

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(),
            "tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier()
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}

    def ajustar_modelo(self, modelo_nombre, param_grid=None, cross_validation = 5, ruta_guardar_modelo = "",nombre_modelo_guardar="mejor_modelo.pkl"):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo 
        parametros_default = {
            "logistic_regression": {
                'penalty': ['l1', 'l2', 'elasticnet','none'], 
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            },
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Ajuste del modelo
        grid_search = GridSearchCV(estimator=modelo, 
                                   param_grid=param_grid, 
                                   cv=cross_validation, 
                                   scoring='accuracy')
        
        grid_search.fit(self.X_train, self.y_train)
        self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
        self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)
        self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_
        # Guardar el modelo
        with open(f'{ruta_guardar_modelo}/{nombre_modelo_guardar}', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

    def calcular_metricas(self, modelo_nombre):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC y Kappa.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        # Calcular probabilidades para AUC (si el modelo las soporta)
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if hasattr(modelo, "predict_proba"):
            self.prob_train = modelo.predict_proba(self.X_train)[:, 1]
            self.prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            self.prob_train = self.prob_test = None  # Si no hay probabilidades, AUC no será calculado

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, self.prob_train) if self.prob_train is not None else None
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, self.prob_test) if self.prob_test is not None else None
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test})

    def plot_matriz_confusion(self, modelo_nombre, invertir=True, tamano_grafica=(4, 3), labels=False, label0="", label1=""):
        """
        Genera un heatmap para visualizar una matriz de confusión.

        Args:
            matriz_confusion (numpy.ndarray): Matriz de confusión que se desea graficar.
            invertir (bool, opcional): Si es True, invierte el orden de las filas y columnas de la matriz
                para reflejar el eje Y invertido (orden [1, 0] en lugar de [0, 1]). Por defecto, True.
            tamano_grafica (tuple, opcional): Tamaño de la figura en pulgadas. Por defecto, (4, 3).
            labels (bool, opcional): Si es True, permite agregar etiquetas personalizadas a las clases
                utilizando `label0` y `label1`. Por defecto, False.
            label0 (str, opcional): Etiqueta personalizada para la clase 0 (negativa). Por defecto, "".
            label1 (str, opcional): Etiqueta personalizada para la clase 1 (positiva). Por defecto, "".

        Returns:
            None: La función no retorna ningún valor, pero muestra un heatmap con la matriz de confusión.

            Ejemplos:
        > from sklearn.metrics import confusion_matrix
        >         >>> y_true = [0, 1, 1, 0, 1, 1]
        >         >>> y_pred = [0, 1, 1, 0, 0, 1]
        >         >>> matriz_confusion = confusion_matrix(y_true, y_pred)
        >         >>> plot_matriz_confusion(matriz_confusion, invertir=True, labels=True,label0="Clase Negativa", label1="Clase Positiva")
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        matriz_confusion = confusion_matrix(self.y_test, pred_test)
        if invertir == True:
            plt.figure(figsize=(tamano_grafica))
            if labels == True:
                labels = [f'1: {label1}', f'0: {label0}']
            else:
                labels = [f'1', f'0']
            sns.heatmap(matriz_confusion[::-1, ::-1], annot=True, fmt="g",cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.title("Matriz de Confusión (Invertida)")
        else: 
            plt.figure(figsize=(tamano_grafica))
            if labels == True:
                labels = [f'0: {label0}', f'1: {label1}']
            else:
                labels = [f'0', f'1']
            sns.heatmap(matriz_confusion, annot=True, fmt="g",cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.title("Matriz de Confusión")

    def plot_curva_ROC(self, grafica_size = (9,7)):
        """
        Plotea la curva ROC utilizando las probabilidades calculadas.
        """
        if self.prob_test is None:
            raise ValueError("Debe calcular las probabilidades (calcular_metricas) antes de graficar la curva ROC.")
        
        fpr, tpr, _ = roc_curve(self.y_test, self.prob_test)
        plt.figure(figsize=grafica_size)
        sns.lineplot(x=fpr, y=tpr, color="orange", label="Modelo")
        sns.lineplot(x=[0, 1], y=[0, 1], color="grey", linestyle="--", label="Azar")
        plt.xlabel("1 - Especificidad (Ratio Falsos Positivos)")
        plt.ylabel("Recall (Ratio Verdaderos Positivos)")
        plt.title("Curva ROC")
        plt.legend()
        plt.show()
    
    def importancia_predictores(self, modelo_nombre):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()