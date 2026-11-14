
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocesamiento import spacy_preprocessor


# 1. CARGAR EL DATASET

df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

resenas = df["Review"].dropna().astype(str)

print("=== CARGA DE DATOS ===")
print(f"Total de reseñas: {len(resenas)}")


# 2. NORMALIZACIÓN Y LEMATIZACIÓN

print("\nProcesando reseñas con spaCy...")

corpus_lematizado = [
    spacy_preprocessor(resena)
    for resena in resenas
]

corpus_lematizado = [
    texto for texto in corpus_lematizado
    if texto.strip()
]

print(f"Documentos procesados: {len(corpus_lematizado)}")


# 3. BAG OF WORDS

bow_vectorizer = CountVectorizer()

X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

vocabulario_bow = bow_vectorizer.get_feature_names_out()

print("\n=== BAG OF WORDS ===")
print(f"Dimensiones: {X_bow.shape}")
print(f"Tamaño del vocabulario: {len(vocabulario_bow)}")
print(f"Valores distintos de cero: {X_bow.nnz}")


# 4. TF-IDF

tfidf_vectorizer = TfidfVectorizer()

X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

vocabulario_tfidf = tfidf_vectorizer.get_feature_names_out()

print("\n=== TF-IDF ===")
print(f"Dimensiones: {X_tfidf.shape}")
print(f"Tamaño del vocabulario: {len(vocabulario_tfidf)}")
print(f"Valores distintos de cero: {X_tfidf.nnz}")


# 5. COMPARACIÓN DE RESULTADOS

print("\n=== COMPARACIÓN ===")

print("Primera reseña normalizada:")
print(corpus_lematizado[0])

print("\nBag-of-Words:")
print("Frecuencias:", X_bow[0].toarray()[0][:15])

print("\nTF-IDF:")
print("Pesos:", X_tfidf[0].toarray()[0][:15])


# 6. PALABRAS MÁS RELEVANTES SEGÚN TF-IDF

pesos = X_tfidf[0].toarray().flatten()

indices = pesos.argsort()[-10:][::-1]

print("\n=== 10 PALABRAS MÁS RELEVANTES ===")

for indice in indices:
    if pesos[indice] > 0:
        print(
            f"{vocabulario_tfidf[indice]}: "
            f"{pesos[indice]:.4f}"
        )


print("\nVectorización completada correctamente.")


# 7. VISUALIZACIÓN VECTORIAL EN 3D

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def graficar_documentos_3d(ax, matriz, titulo, color):
    # Reducir las características a tres dimensiones.
    modelo = TruncatedSVD(n_components=3, random_state=42)
    coordenadas = modelo.fit_transform(matriz)

    x = coordenadas[:, 0]
    y = coordenadas[:, 1]
    z = coordenadas[:, 2]

    ax.scatter(
        x, y, z,
        c=color,
        s=15,
        alpha=0.6
    )

    ax.set_title(titulo)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_zlabel("Componente 3")


fig = plt.figure(figsize=(16, 7))

# Gráfica Bag-of-Words
ax1 = fig.add_subplot(121, projection="3d")

graficar_documentos_3d(
    ax1,
    X_bow,
    "Espacio vectorial BoW",
    "orange"
)

# Gráfica TF-IDF
ax2 = fig.add_subplot(122, projection="3d")

graficar_documentos_3d(
    ax2,
    X_tfidf,
    "Espacio vectorial TF-IDF",
    "teal"
)

plt.tight_layout()

# Guardar evidencia para la entrega.
plt.savefig("comparacion_vectorizacion.png", dpi=200)

plt.show()
