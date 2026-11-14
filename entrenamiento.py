
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from preprocesamiento import spacy_preprocessor

# 1. Ingesta de datos
df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

X = df["Review"]
y = df["Liked"]

# 2. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear pipeline
pipeline_nlp = Pipeline([
    ("vectorizacion", TfidfVectorizer(
        preprocessor=spacy_preprocessor
    )),
    ("modelado", MultinomialNB())
])

# 4. Entrenar
print("Entrenando modelo...")
pipeline_nlp.fit(X_train, y_train)

# 5. Evaluar
print("\nEvaluando modelo...")
predicciones = pipeline_nlp.predict(X_test)

print(classification_report(y_test, predicciones))

# 6. Probar nuevas reseñas
resenas = [
    "The food was absolutely amazing and tasty",
    "The service was terrible",
    "I really loved this restaurant",
    "The food was cold and disgusting"
]

resultados = pipeline_nlp.predict(resenas)

for resena, resultado in zip(resenas, resultados):
    sentimiento = "Positiva" if resultado == 1 else "Negativa"
    print(f"Reseña: {resena}")
    print(f"Predicción: {sentimiento}\n")

# 7. Guardar modelo entrenado
joblib.dump(pipeline_nlp, "pipeline.pkl")

print("Modelo guardado correctamente en pipeline.pkl")
