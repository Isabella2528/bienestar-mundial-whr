"""
analisis.py
Funciones de análisis (estandarización, ACP, K-Means y perfilamiento).
No depende de Streamlit: se puede probar y reutilizar fuera de la app.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
N_INIT = 20

VARS_MODELO = ["pib_pc_log", "apoyo_social", "esperanza_vida_sana",
               "libertad", "generosidad", "percepcion_corrupcion"]
VARS_SUPLEMENTARIAS = ["evaluacion_vida", "distopia_residual"]

ETIQUETAS = {
    "pib_pc_log": "PIB per cápita (log)",
    "apoyo_social": "Apoyo social",
    "esperanza_vida_sana": "Esperanza de vida saludable",
    "libertad": "Libertad para decidir",
    "generosidad": "Generosidad",
    "percepcion_corrupcion": "Percepción de corrupción",
    "evaluacion_vida": "Evaluación de vida (puntaje)",
    "distopia_residual": "Distopía + residual",
}


def estandarizar(df, variables):
    """Z-score por variable (media 0, desviación 1) dentro de la muestra recibida."""
    escalador = StandardScaler()
    Z = escalador.fit_transform(df[variables])
    return Z, escalador


def ejecutar_acp(Z, variables):
    """ACP sobre la matriz estandarizada. Devuelve tabla de varianza, cargas y puntuaciones."""
    pca = PCA(random_state=SEMILLA).fit(Z)
    componentes = pca.components_.copy()
    puntuaciones = pca.transform(Z)

    # Los signos de un componente son arbitrarios: se fijan para que la carga
    # de mayor magnitud sea positiva (resultados reproducibles).
    signos = np.sign(componentes[np.arange(len(componentes)),
                                 np.abs(componentes).argmax(axis=1)])
    componentes *= signos[:, None]
    puntuaciones *= signos

    eig = pca.explained_variance_
    razon = pca.explained_variance_ratio_
    nombres = [f"CP{i + 1}" for i in range(len(variables))]

    tabla = pd.DataFrame({
        "Componente": nombres,
        "Autovalor": eig,
        "Varianza explicada (%)": razon * 100,
        "Varianza acumulada (%)": razon.cumsum() * 100,
    })
    # Cargas = correlación entre cada variable y cada componente
    cargas = pd.DataFrame(componentes.T * np.sqrt(eig), index=variables, columns=nombres)
    return {
        "tabla": tabla,
        "cargas": cargas,
        "puntuaciones": pd.DataFrame(puntuaciones, columns=nombres),
        "autovalores": eig,
        "razon": razon,
    }


def componentes_sugeridos(autovalores, razon):
    """Criterios de retención: Kaiser (autovalor > 1) y varianza acumulada >= 80 %."""
    kaiser = int((autovalores > 1).sum())
    n80 = int(np.argmax(razon.cumsum() >= 0.80) + 1)
    return kaiser, n80


def evaluar_k(X, ks, semilla=SEMILLA):
    """Inercia (codo) y coeficiente de silueta para cada K."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=semilla).fit(X)
        filas.append({"K": k, "Inercia": km.inertia_,
                      "Silueta": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas)


def ajustar_kmeans(X, k, semilla=SEMILLA):
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=semilla).fit(X)
    return km.labels_, km


def nombrar_grupos(etiquetas):
    """
    Asigna letras (Grupo A, B, C...) según el tamaño del grupo, de mayor a menor.
    La letra NO implica orden, jerarquía ni calidad: solo identifica el grupo.
    """
    tamanos = pd.Series(etiquetas).value_counts()
    orden = sorted(tamanos.index, key=lambda c: (-tamanos[c], c))
    letras = {c: f"Grupo {chr(65 + i)}" for i, c in enumerate(orden)}
    return pd.Series([letras[c] for c in etiquetas])


def silueta_por_observacion(X, etiquetas):
    return silhouette_samples(X, etiquetas)


def perfil_grupos(df, grupos, columnas):
    """Medias por grupo en unidades originales, más el tamaño del grupo."""
    g = df[columnas].groupby(grupos.values)
    perfil = g.mean()
    perfil.insert(0, "n países", g.size())
    return perfil.sort_index()


def perfil_relativo(df, grupos, columnas):
    """Media de cada grupo expresada en desviaciones estándar respecto a la media global."""
    media, desv = df[columnas].mean(), df[columnas].std(ddof=0)
    return ((df[columnas].groupby(grupos.values).mean() - media) / desv).sort_index()
