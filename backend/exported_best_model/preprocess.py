
import pandas as pd
import numpy as np
import joblib
import json

def preprocess_input(raw_df, scaler, top5):
    """
    raw_df: DataFrame con las 18 columnas originales (mismos nombres que en entrenamiento).
    scaler: StandardScaler ajustado.
    top5: lista de las 5 características seleccionadas.
    Retorna un DataFrame con las 28 columnas en el orden correcto para el modelo.
    """
    continuous_cols = ['ENERGY_LEVEL', 'OXYGEN_SATURATION']
    df = raw_df.copy()
    # Escalar columnas continuas
    for col in continuous_cols:
        df[col] = scaler.transform(df[[col]]).ravel()
    # Crear interacciones multiplicativas entre las top5
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            col_name = f'{top5[i]}_x_{top5[j]}'
            df[col_name] = df[top5[i]] * df[top5[j]]
    # Reordenar según el orden esperado (leer desde feature_names.json)
    with open('feature_names.json', 'r') as f:
        expected_order = json.load(f)
    for col in expected_order:
        if col not in df.columns:
            df[col] = 0
    return df[expected_order]
