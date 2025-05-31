import joblib
import pandas as pd

# Definimos un nuevo caso de prueba
nuevo_caso = pd.DataFrame({
    'temperature': [25],          # Temperatura en °C
    'flyers': [50],               # Número de flyers repartidos
    'month': [7],                 # Mes (ej. 7 = julio)
    'day_Friday': [0],            # No es viernes
    'day_Monday': [1],            # Es lunes (1 = sí, 0 = no)
    'day_Saturday': [0],
    'day_Sunday': [0],
    'day_Thursday': [0],
    'day_Tuesday': [0],
    'day_Wednesday': [0],
    'rainfall_heavy rain': [0],   # No lluvia fuerte
    'rainfall_light rain': [1],   # Lluvia ligera (1 = sí)
    'rainfall_sunny': [0]         # No está soleado
})

# Aseguramos el orden de columnas (usando X.columns del modelo)
nuevo_caso = nuevo_caso[X.columns]  

# Realizamos la predicción y convertimos a entero
prediccion = int(modelo_regresion.predict(nuevo_caso)[0])  # Conversión directa a int

# Imprimimos el resultado como número entero
print(f'El número de ventas estimado para esos datos es: {prediccion}') 
