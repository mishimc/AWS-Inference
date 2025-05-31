from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar el modelo de regresión (ajusta la ruta según tu estructura)
model = joblib.load("final_regresion_model.pkl")

# Variables necesarias para el orden de columnas (simula tu X.columns)
# Esto es un ejemplo: reemplázalo con las columnas REALES de tu modelo
expected_columns = [
    'temperature', 'flyers', 'month',
    'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday',
    'day_Thursday', 'day_Tuesday', 'day_Wednesday',
    'rainfall_heavy rain', 'rainfall_light rain', 'rainfall_sunny'
]

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("forms.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    temperature: float = Form(...),
    flyers: int = Form(...),
    month: int = Form(...),
    day: str = Form(...),  # Día de la semana (ej: "Monday")
    rainfall: str = Form(...)  # Condición climática (ej: "light_rain")
):
    # Mapear día seleccionado a one-hot encoding
    day_columns = {
        'Friday': [1, 0, 0, 0, 0, 0, 0],
        'Monday': [0, 1, 0, 0, 0, 0, 0],
        'Saturday': [0, 0, 1, 0, 0, 0, 0],
        'Sunday': [0, 0, 0, 1, 0, 0, 0],
        'Thursday': [0, 0, 0, 0, 1, 0, 0],
        'Tuesday': [0, 0, 0, 0, 0, 1, 0],
        'Wednesday': [0, 0, 0, 0, 0, 0, 1],
    }
    
    # Mapear clima a one-hot encoding
    rainfall_columns = {
        'heavy_rain': [1, 0, 0],
        'light_rain': [0, 1, 0],
        'sunny': [0, 0, 1],
    }

    # Crear DataFrame con los datos del formulario
    features = pd.DataFrame({
        'temperature': [temperature],
        'flyers': [flyers],
        'month': [month],
        'day_Friday': [day_columns[day][0]],
        'day_Monday': [day_columns[day][1]],
        'day_Saturday': [day_columns[day][2]],
        'day_Sunday': [day_columns[day][3]],
        'day_Thursday': [day_columns[day][4]],
        'day_Tuesday': [day_columns[day][5]],
        'day_Wednesday': [day_columns[day][6]],
        'rainfall_heavy rain': [rainfall_columns[rainfall][0]],
        'rainfall_light rain': [rainfall_columns[rainfall][1]],
        'rainfall_sunny': [rainfall_columns[rainfall][2]],
    })

    # Asegurar el orden correcto de columnas
    features = features[expected_columns]

    # Predecir
    prediction = int(model.predict(features)[0])

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "result": f"Ventas estimadas: {prediction} unidades",
        },
    )
