import pickle
import pandas as pd
import numpy as np

# Cargar los modelos y transformadores entrenados
def load_models():
    """
    Carga modelos y transformadores previamente entrenados desde archivos pickle.

    Retorna:
    --------
    tuple
        Una tupla que contiene los siguientes objetos:
            - target_encoder: Codificador para transformar la variable objetivo.
            - one_hot_encoder: Codificador One-Hot para variables categóricas.
            - robust_scaler: Escalador robusto para normalizar los datos.
            - modelo: Modelo de aprendizaje automático (XGBoost en este caso).

    Notas:
    ------
    - Los archivos deben estar ubicados en la carpeta `../datos/modelos-encoders/`.
    - Se espera que los archivos tengan los nombres:
        - `target_encoder.pkl`
        - `one_hot_encoder.pkl`
        - `robust_scaler.pkl`
        - `modelo_xgb.pkl`

    Ejemplo:
    --------
    target_encoder, one_hot_encoder, robust_scaler, modelo = load_models()
    """

    with open('../datos/modelos-encoders/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('../datos/modelos-encoders/one_hot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)
    with open('../datos/modelos-encoders/robust_scaler.pkl', 'rb') as f:
        robust_scaler = pickle.load(f)    
    with open('../datos/modelos-encoders/modelo_xgb.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return target_encoder, one_hot_encoder, robust_scaler, modelo

def load_options():
    with open('../datos/lista_opciones/01_tipo_propiedad.pkl', 'rb') as archivo:
        lista_tipo_propiedad = pickle.load(archivo)

    with open('../datos/lista_opciones/02_num_habitaciones.pkl', 'rb') as archivo:
        lista_habitaciones = pickle.load(archivo)

    with open('../datos/lista_opciones/03_num_aseos.pkl', 'rb') as archivo:
        lista_aseos = pickle.load(archivo)

    with open('../datos/lista_opciones/04_floor.pkl', 'rb') as archivo:
        lista_floor = pickle.load(archivo)

    with open('../datos/lista_opciones/05_municipio.pkl', 'rb') as archivo:
        lista_municipality = pickle.load(archivo)

    with open('../datos/lista_opciones/06_distrito.pkl', 'rb') as archivo:
        lista_district = pickle.load(archivo)

    with open('../datos/lista_opciones/07_ascensor.pkl', 'rb') as archivo:
        lista_ascensor = pickle.load(archivo)

    with open('../datos/lista_opciones/08_distancia_centro.pkl', 'rb') as archivo:
        lista_distancia = pickle.load(archivo)
    
    return lista_tipo_propiedad, lista_habitaciones, lista_aseos, lista_floor, lista_municipality, lista_district, lista_ascensor, lista_distancia

def realizar_prediccion(tipo_propiedad, prop_size, habitaciones, aseos, prop_floor, municipio, distrito, ascensor, dist_centro,
                        encoder_ordinales, encoder_nominales, scaler, modelo
                        ):
    """
    Realiza una predicción del precio de una propiedad basada en sus características utilizando un modelo entrenado.

    Parámetros:
    -----------
    tipo_propiedad : str
        Tipo de propiedad (e.g., "piso", "chalet", etc.).
    prop_size : float
        Tamaño de la propiedad en metros cuadrados.
    habitaciones : int
        Número de habitaciones de la propiedad.
    aseos : int
        Número de aseos en la propiedad.
    prop_floor : int
        Número de planta en la que se encuentra la propiedad.
    municipio : str
        Nombre del municipio donde se encuentra la propiedad.
    distrito : str
        Nombre del distrito donde se encuentra la propiedad.
    ascensor : int
        Indicador de si la propiedad tiene ascensor (1 para sí, 0 para no).
    dist_centro : float
        Distancia desde el centro de la ciudad en kilómetros.
    encoder_ordinales : object
        Codificador para variables ordinales (Target Encoder).
    encoder_nominales : object
        Codificador para variables nominales (OneHotEncoder).
    scaler : object
        Escalador para normalizar los datos (e.g., RobustScaler).
    modelo : object
        Modelo de aprendizaje automático entrenado para realizar la predicción.

    Retorna:
    --------
    numpy.ndarray
        Predicción del modelo para el precio de la propiedad.

    Notas:
    ------
    - Las columnas se dividen en tres categorías:
        - `cols_target`: Variables que serán codificadas con el Target Encoder.
        - `cols_nominales`: Variables que serán codificadas con el OneHotEncoder.
        - `cols_escalar`: Variables que serán escaladas antes de la predicción.
    - El DataFrame `df_new` contiene las características de la propiedad y pasa por varias transformaciones antes de ser usado en el modelo.

    Ejemplo:
    --------
    prediccion = realizar_prediccion(
        tipo_propiedad="piso",
        prop_size=80,
        habitaciones=3,
        aseos=2,
        prop_floor=5,
        municipio="Madrid",
        distrito="Centro",
        ascensor=1,
        dist_centro=2.5,
        encoder_ordinales=target_encoder,
        encoder_nominales=one_hot_encoder,
        scaler=robust_scaler,
        modelo=modelo_xgb
    )
    print(f"El precio estimado es: {prediccion[0]}")
    """
    
    cols_target = ["bathrooms","municipality","district","distancia_centro"]
    cols_nominales = ["propertyType","rooms","floor","hasLift"]
    cols_escalar = ["size","bathrooms","municipality","district","distancia_centro"]
    # Datos de una nueva casa para predicción
    new_house = pd.DataFrame({
        'propertyType': [tipo_propiedad],  # Nueva categoría no vista
        'size': [prop_size],
        'rooms': [habitaciones],
        'bathrooms': [aseos],
        'floor': [prop_floor],
        'municipality': [municipio],
        'district': [distrito],
        'hasLift' : [ascensor],
        'distancia_centro' : [dist_centro]
    })

    df_new = pd.DataFrame(new_house)
    df_pred = df_new.copy()

    # Hacemos el OneHot Encoder
    onehot = encoder_nominales.transform(df_pred[cols_nominales])
    # Obtenemos los nombres de las columnas del codificador
    column_names = encoder_nominales.get_feature_names_out(cols_nominales)
    # Convertimos a un DataFrame
    onehot_df = pd.DataFrame(onehot.toarray(), columns=column_names)

    # Realizamos el target encoder
    df_pred["precio"] = np.nan #La creo porque la espera, luego se borra
    df_pred = encoder_ordinales.transform(df_pred)

    # Quitamos las columnas que ya han sido onehoteadas 
    df_pred.drop(columns= cols_nominales,inplace=True)
    df_pred = pd.concat([df_pred, onehot_df], axis=1)

    # Escalamos los valores
    df_pred[cols_escalar] = scaler.transform(df_pred[cols_escalar])

    # Realizamos la predicción
    df_pred.drop(columns="precio",inplace=True)
    prediccion = modelo.predict(df_pred)
    return prediccion