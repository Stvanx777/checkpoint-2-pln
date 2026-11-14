
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Importar preprocesamiento para cargar el pipeline
from preprocesamiento import spacy_preprocessor

app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="Clasificación de reseñas de restaurantes",
    version="1.0"
)

# Cargar modelo entrenado
modelo = joblib.load("pipeline.pkl")

# Definir estructura de entrada
class Resena(BaseModel):
    texto: str

# Ruta principal
@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente"}

# Ruta de predicción
@app.post("/predict")
def predecir(resena: Resena):
    resultado = modelo.predict([resena.texto])[0]

    sentimiento = "Positiva" if resultado == 1 else "Negativa"

    return {
        "texto": resena.texto,
        "prediccion": sentimiento,
        "valor": int(resultado)
    }
