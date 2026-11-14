"""
app.py
¿Qué dimensiones y perfiles de bienestar distinguen a los países?
ACP y K-Means sobre los factores del World Happiness Report 2025.

Ejecutar:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analisis import (ETIQUETAS, VARS_MODELO, VARS_SUPLEMENTARIAS, ajustar_kmeans,
                          componentes_sugeridos, ejecutar_acp, estandarizar, evaluar_k,
                          nombrar_grupos, perfil_grupos, perfil_relativo,
                          silueta_por_observacion)

st.set_page_config(page_title="Bienestar mundial: ACP y K-Means", page_icon="🌍", layout="wide")

BASE = Path(__file__).parent
RUTA_DATOS = BASE / "data" / "processed" / "whr_limpio.csv"
RUTA_REPORTE = BASE / "data" / "processed" / "reporte_limpieza.md"

KS = tuple(range(2, 9))
ENTRADA_CP = "Componentes principales seleccionados"
ENTRADA_VAR = "Variables estandarizadas (las 6)"
COLS_PERFIL = VARS_MODELO + VARS_SUPLEMENTARIAS


def et(variable):
    return ETIQUETAS.get(variable, variable)


# ----------------------------------------------------------------------
# Carga y cálculos con caché (la depuración ya se hizo en src/limpieza.py)
# ----------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


@st.cache_data(show_spinner=False)
def evaluar_k_cache(X, ks):
    return evaluar_k(X, list(ks))


datos = cargar_datos()
anios = sorted(datos["anio"].unique().tolist(), reverse=True)

# Selecciones del usuario que deben sobrevivir al cambio de sección
cfg = st.session_state.setdefault("cfg", {})

# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
st.sidebar.title("🌍 Bienestar mundial")
pagina = st.sidebar.radio(
    "Sección",
    ["1. Contexto y datos", "2. Exploración", "3. ACP", "4. K-Means", "5. Discusión"],
)
anio = st.sidebar.selectbox(
    "Año de análisis", anios, index=0,
    help="Cada análisis usa un solo año: así cada país aparece una vez y la estandarización "
         "compara a los países con sus pares del mismo año.")
st.sidebar.caption("Fuente: World Happiness Report 2025 (Data for Figure 2.1). "
                   "Datos ya depurados con `src/limpieza.py`.")

df = datos[datos["anio"] == anio].reset_index(drop=True)
Z, _ = estandarizar(df, VARS_MODELO)
acp = ejecutar_acp(Z, VARS_MODELO)
kaiser, n80 = componentes_sugeridos(acp["autovalores"], acp["razon"])
todas_cp = list(acp["tabla"]["Componente"])


# ----------------------------------------------------------------------
# Utilidades compartidas
# ----------------------------------------------------------------------
def slider_componentes():
    n = st.slider("Número de componentes a conservar", 1, len(VARS_MODELO),
                  value=int(cfg.get("n_comp", kaiser)),
                  help="Por defecto, el criterio de Kaiser (autovalor > 1).")
    cfg["n_comp"] = n
    return n


def matriz_entrada(n_comp, entrada):
    if entrada == ENTRADA_CP:
        return acp["puntuaciones"].iloc[:, :n_comp].to_numpy()
    return Z


def resolver_k(n_comp, entrada, ev):
    """K sugerido (máxima silueta) salvo que el usuario ya eligió otro para esta misma configuración."""
    k_sug = int(ev.loc[ev["Silueta"].idxmax(), "K"])
    ctx = (int(anio), entrada, int(n_comp) if entrada == ENTRADA_CP else 0)
    k = cfg["k"] if (cfg.get("k_ctx") == ctx and "k" in cfg) else k_sug
    return k, k_sug, ctx


# ----------------------------------------------------------------------
# 1. Contexto y datos
# ----------------------------------------------------------------------
def pagina_contexto():
    st.title("¿Qué dimensiones y perfiles de bienestar distinguen a los países?")
    st.markdown(
        "Este proyecto aplica **Análisis de Componentes Principales (ACP)** y **K-Means** a los "
        "factores con los que el *World Happiness Report 2025* explica la evaluación de vida de "
        "cada país. La idea es pasar de una tabla de seis indicadores por país a dos lecturas: "
        "qué dimensiones resumen las diferencias entre países y si existen perfiles de países "
        "con una combinación parecida de condiciones."
    )

    st.subheader("Qué representa cada observación")
    st.markdown(
        f"Cada fila es **un país en un año**. El análisis principal usa un solo año a la vez "
        f"(ahora: **{anio}**, con **{len(df)} países**) para que cada país aparezca una sola vez. "
        "Puedes cambiar el año en la barra lateral."
    )

    st.subheader("Fuente, cobertura y variables")
    st.markdown(
        "- **Fuente:** *World Happiness Report 2025*, archivo *Data for Figure 2.1*. El informe se "
        "basa en la Gallup World Poll.\n"
        "- **Cobertura:** unos 140-150 países por año; los seis factores están disponibles de 2019 "
        "a 2024 (antes solo hay puntaje y ranking).\n"
        "- **La evaluación de vida** es un promedio de tres años de la escala de 0 a 10 con la que "
        "las personas valoran su vida."
    )
    variables = pd.DataFrame({
        "Variable": [et(v) for v in COLS_PERFIL],
        "Rol en el análisis": ["Entra a ACP y K-Means"] * 6
        + ["Suplementaria: no entra al modelo, sirve para interpretar y validar"] * 2,
        "Qué es": [
            "Contribución del PIB per cápita (en logaritmo) a la evaluación de vida",
            "Contribución del apoyo social",
            "Contribución de la esperanza de vida saludable",
            "Contribución de la libertad para tomar decisiones de vida",
            "Contribución de la generosidad",
            "Contribución de la percepción de corrupción (más alto = menos corrupción percibida)",
            "Puntaje de evaluación de vida (0 a 10)",
            "Parte del puntaje que los seis factores no explican",
        ],
    })
    st.dataframe(variables, hide_index=True)
    st.caption("Los seis factores son contribuciones estimadas por el modelo del informe, medidas en "
               "puntos de la escala de vida; no son los datos crudos de cada indicador.")

    st.subheader("Por qué ACP y K-Means")
    st.markdown(
        "- **ACP:** los factores están correlacionados entre sí (por ejemplo, PIB, esperanza de vida "
        "y apoyo social suelen moverse juntos). El ACP resume esa información compartida en pocas "
        "dimensiones que se pueden dibujar e interpretar.\n"
        "- **K-Means:** un ranking de un solo número oculta que países con puntajes parecidos pueden "
        "tener combinaciones distintas de factores. K-Means permite explorar si hay perfiles de "
        "países y describirlos con las variables originales."
    )

    st.subheader("Limitaciones de la fuente")
    st.markdown(
        "- Son contribuciones modeladas, no mediciones directas.\n"
        "- Los datos vienen de encuestas con muestras limitadas por país y no cubren todos los países.\n"
        "- Es una foto por año: no permite hablar de causalidad."
    )

    st.subheader("Preparación de los datos")
    st.markdown(
        "La base se depuró **antes** de la app con `src/limpieza.py`: se dejaron solo los años con "
        "los seis factores, se eliminaron las filas incompletas, se verificó la ausencia de duplicados "
        "y de valores fuera de rango, y se comprobó que el puntaje es la suma de sus componentes "
        "(por eso no entra al modelo). Los valores extremos se conservan porque son países reales."
    )
    if RUTA_REPORTE.exists():
        with st.expander("Ver la bitácora completa de depuración"):
            st.markdown(RUTA_REPORTE.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# 2. Exploración
# ----------------------------------------------------------------------
def pagina_exploracion():
    st.header("2. Exploración de datos (base ya depurada)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Países", len(df))
    c2.metric("Variables del modelo", len(VARS_MODELO))
    c3.metric("Valores faltantes", int(df[COLS_PERFIL].isna().sum().sum()))
    c4.metric("Año", int(anio))

    st.subheader("Estructura de la base")
    estructura = pd.DataFrame({
        "Variable": [et(v) for v in COLS_PERFIL],
        "Tipo": [str(df[v].dtype) for v in COLS_PERFIL],
        "Rol": ["Modelo"] * len(VARS_MODELO) + ["Suplementaria"] * len(VARS_SUPLEMENTARIAS),
    })
    st.dataframe(estructura, hide_index=True)

    st.subheader("Estadísticas descriptivas")
    desc = df[COLS_PERFIL].describe().T.rename(index=ETIQUETAS).round(3)
    st.dataframe(desc)
    st.caption("Las escalas difieren (el PIB llega cerca de 2 y la generosidad a 0,3), por eso se "
               "estandariza antes del ACP: así ninguna variable domina solo por su magnitud.")

    st.subheader("Distribución de una variable")
    var = st.selectbox("Variable", COLS_PERFIL, format_func=et)
    col1, col2 = st.columns(2)
    fig_h = px.histogram(df, x=var, nbins=25, labels={var: et(var)},
                         title=f"Distribución de {et(var)} ({anio})")
    fig_h.update_layout(yaxis_title="Número de países")
    col1.plotly_chart(fig_h)
    fig_b = px.box(df, y=var, points="all", hover_name="pais", labels={var: et(var)},
                   title=f"Diagrama de caja de {et(var)} ({anio})")
    col2.plotly_chart(fig_b)

    st.subheader("Matriz de correlaciones")
    incluir = st.checkbox("Incluir variables suplementarias", value=False)
    cols = COLS_PERFIL if incluir else VARS_MODELO
    corr = df[cols].corr()
    etq = [et(v) for v in cols]
    fig_c = px.imshow(corr.round(2).values, x=etq, y=etq, color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, text_auto=True, aspect="auto",
                      title=f"Correlación de Pearson entre variables ({anio})")
    st.plotly_chart(fig_c)

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(mask).stack()
    fuerte = pares.idxmax()
    debil = pares.abs().idxmin()
    st.markdown(
        f"**Lectura:** la pareja más correlacionada es *{et(fuerte[0])}* y *{et(fuerte[1])}* "
        f"(r = {pares[fuerte]:.2f}); la más independiente es *{et(debil[0])}* y *{et(debil[1])}* "
        f"(r = {pares[debil]:.2f}). Las correlaciones altas son la razón de ser del ACP: hay "
        "información redundante que se puede resumir. Una variable casi sin correlación con las "
        "demás tenderá a formar su propia dimensión."
    )

    with st.expander("Ver la tabla de datos"):
        st.dataframe(df[["pais", "ranking"] + COLS_PERFIL].rename(columns=ETIQUETAS | {"pais": "País", "ranking": "Ranking"}),
                     hide_index=True)


# ----------------------------------------------------------------------
# 3. ACP
# ----------------------------------------------------------------------
def pagina_acp():
    st.header("3. Análisis de Componentes Principales")
    st.markdown(
        "Antes del ACP, cada variable se **estandariza** (media 0, desviación 1) para que todas "
        "pesen igual. Con datos estandarizados el ACP trabaja sobre la matriz de correlaciones."
    )
    with st.expander("Ver la matriz estandarizada"):
        zdf = pd.DataFrame(Z, columns=[et(v) for v in VARS_MODELO])
        zdf.insert(0, "País", df["pais"])
        st.dataframe(zdf.round(3), hide_index=True)
        st.caption(f"Comprobación: la mayor media absoluta es {np.abs(Z.mean(axis=0)).max():.1e} "
                   f"y todas las desviaciones son {Z.std(axis=0).round(2).min():.2f}.")

    st.subheader("3.1 Varianza explicada")
    tabla = acp["tabla"]
    st.dataframe(tabla.round(2), hide_index=True)

    col1, col2 = st.columns(2)
    f1 = go.Figure(go.Scatter(x=tabla["Componente"], y=tabla["Autovalor"], mode="lines+markers"))
    f1.add_hline(y=1, line_dash="dash", annotation_text="Autovalor = 1 (criterio de Kaiser)")
    f1.update_layout(title="Gráfico de sedimentación", xaxis_title="Componente",
                     yaxis_title="Autovalor (varianza que explica el componente)")
    col1.plotly_chart(f1)

    f2 = go.Figure()
    f2.add_bar(x=tabla["Componente"], y=tabla["Varianza explicada (%)"], name="Individual")
    f2.add_scatter(x=tabla["Componente"], y=tabla["Varianza acumulada (%)"],
                   mode="lines+markers", name="Acumulada")
    f2.add_hline(y=80, line_dash="dash", annotation_text="80 %")
    f2.update_layout(title="Varianza explicada por componente y acumulada",
                     xaxis_title="Componente", yaxis_title="% de la varianza total",
                     yaxis_range=[0, 105])
    col2.plotly_chart(f2)

    st.subheader("3.2 ¿Cuántos componentes conservar?")
    st.markdown(
        f"- **Criterio de Kaiser** (autovalor > 1): **{kaiser}** componente(s).\n"
        f"- **Varianza acumulada de al menos 80 %:** **{n80}** componente(s).\n"
        "- **Codo del gráfico de sedimentación:** busca el punto donde la curva deja de caer de "
        "forma pronunciada."
    )
    n_comp = slider_componentes()
    acum = float(tabla["Varianza acumulada (%)"].iloc[n_comp - 1])
    st.info(f"Con {n_comp} componente(s) se conserva el **{acum:.1f} %** de la varianza original "
            f"y se pierde el {100 - acum:.1f} %. Los criterios no siempre coinciden: la decisión "
            "es un balance entre simplicidad e información.")

    st.subheader("3.3 Cargas: ¿qué significa cada componente?")
    cargas = acp["cargas"]
    etq_v = [et(v) for v in VARS_MODELO]
    fig_l = px.imshow(cargas.round(2).values, x=list(cargas.columns), y=etq_v,
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True,
                      aspect="auto", title="Cargas (correlación entre cada variable y cada componente)")
    st.plotly_chart(fig_l)

    comp = st.selectbox("Componente a interpretar", todas_cp, index=0)
    serie = cargas[comp].sort_values()
    fig_b = px.bar(x=serie.values, y=[et(v) for v in serie.index], orientation="h",
                   labels={"x": "Carga", "y": ""}, title=f"Cargas sobre {comp}")
    fig_b.update_layout(xaxis_range=[-1, 1])
    st.plotly_chart(fig_b)
    fuertes = cargas[comp][cargas[comp].abs() >= 0.5].sort_values(key=lambda s: -s.abs())
    if len(fuertes):
        texto = ", ".join(f"{et(v)} ({c:+.2f})" for v, c in fuertes.items())
        st.markdown(f"**Variables con |carga| ≥ 0,5 en {comp}:** {texto}. "
                    "Cargas del mismo signo se mueven juntas; de signo contrario, en sentido opuesto.")
    else:
        st.markdown(f"Ninguna variable alcanza |carga| ≥ 0,5 en {comp}.")

    st.subheader("3.4 Los países en el espacio de componentes")
    c1, c2, c3 = st.columns(3)
    cx = c1.selectbox("Eje horizontal", todas_cp, index=0)
    cy = c2.selectbox("Eje vertical", todas_cp, index=1)
    flechas = c3.checkbox("Mostrar las variables como flechas", value=True)
    pct = acp["razon"][todas_cp.index(cx)] * 100 + acp["razon"][todas_cp.index(cy)] * 100
    pts = acp["puntuaciones"][[cx, cy]].copy()
    pts["País"] = df["pais"]
    pts["Evaluación de vida"] = df["evaluacion_vida"]
    fig_s = px.scatter(pts, x=cx, y=cy, color="Evaluación de vida", hover_name="País",
                       color_continuous_scale="Viridis",
                       title=f"Países en el plano {cx}–{cy}: esta vista resume el {pct:.1f} % de la varianza")
    if flechas:
        esc = 0.8 * np.abs(pts[[cx, cy]].to_numpy()).max() / max(np.abs(cargas[[cx, cy]].to_numpy()).max(), 1e-9)
        for v in VARS_MODELO:
            fig_s.add_trace(go.Scatter(
                x=[0, cargas.loc[v, cx] * esc], y=[0, cargas.loc[v, cy] * esc],
                mode="lines+text", text=["", et(v)], textposition="top center",
                line=dict(color="gray", width=1.5), showlegend=False, hoverinfo="skip"))
    st.plotly_chart(fig_s)
    st.caption("El color es la evaluación de vida, que no participó en el ACP: permite ver si los ejes "
               "se relacionan con el bienestar medido. Las flechas muestran hacia dónde crece cada variable.")


# ----------------------------------------------------------------------
# 4. K-Means
# ----------------------------------------------------------------------
def pagina_kmeans():
    st.header("4. Segmentación con K-Means")
    st.markdown(
        "K-Means agrupa los países por cercanía. Puedes aplicarlo sobre los componentes "
        "principales (reduce ruido y es lo que enseñan las gráficas del ACP) o sobre las seis "
        "variables estandarizadas."
    )
    entrada = st.radio("Datos de entrada de K-Means", [ENTRADA_CP, ENTRADA_VAR], horizontal=True,
                       index=0 if cfg.get("entrada", ENTRADA_CP) == ENTRADA_CP else 1)
    cfg["entrada"] = entrada
    n_comp = slider_componentes() if entrada == ENTRADA_CP else int(cfg.get("n_comp", kaiser))
    X = matriz_entrada(n_comp, entrada)
    if entrada == ENTRADA_CP:
        st.caption(f"K-Means usa {n_comp} componente(s), que conservan "
                   f"{acp['razon'][:n_comp].sum() * 100:.1f} % de la varianza.")

    st.subheader("4.1 ¿Cuántos grupos? Codo y silueta")
    ev = evaluar_k_cache(X, KS)
    col1, col2 = st.columns(2)
    f_codo = px.line(ev, x="K", y="Inercia", markers=True,
                     title="Método del codo", labels={"Inercia": "Inercia (suma de distancias² al centro)"})
    col1.plotly_chart(f_codo)
    f_sil = px.line(ev, x="K", y="Silueta", markers=True,
                    title="Coeficiente de silueta promedio", labels={"Silueta": "Silueta promedio"})
    col2.plotly_chart(f_sil)

    k, k_sug, ctx = resolver_k(n_comp, entrada, ev)
    st.markdown(
        f"- **Codo:** busca el K donde la inercia deja de caer con fuerza; es un criterio visual.\n"
        f"- **Silueta:** mide qué tan bien separado está cada país de los otros grupos "
        f"(cerca de 1: bien separado; cerca de 0: en la frontera; negativo: probablemente mal asignado). "
        f"La máxima silueta promedio ocurre en **K = {k_sug}** ({ev.loc[ev['K'] == k_sug, 'Silueta'].iloc[0]:.3f})."
    )
    k = st.slider("K elegido", min(KS), max(KS), value=int(k))
    cfg["k"], cfg["k_ctx"] = k, ctx

    labels, km = ajustar_kmeans(X, k)
    grupos = nombrar_grupos(labels)
    orden = sorted(grupos.unique())
    s = silueta_por_observacion(X, labels)
    st.info("Las letras (Grupo A, B, C…) solo identifican a los grupos, ordenados por tamaño. "
            "No implican jerarquía ni calidad: no interpretes la etiqueta como un orden.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Grupos (K)", k)
    m2.metric("Silueta promedio", f"{s.mean():.3f}")
    m3.metric("Países con silueta < 0,10", int((s < 0.10).sum()))

    st.subheader("4.2 Visualización de los grupos")
    pts = acp["puntuaciones"][["CP1", "CP2"]].copy()
    pts["País"], pts["Grupo"] = df["pais"], grupos
    pct = acp["razon"][:2].sum() * 100
    fig = px.scatter(pts, x="CP1", y="CP2", color="Grupo", hover_name="País",
                     category_orders={"Grupo": orden},
                     title=f"Grupos en el plano CP1–CP2: esta vista resume el {pct:.1f} % de la varianza")
    if entrada == ENTRADA_CP and n_comp >= 2:
        mapa = dict(zip(labels, grupos))
        for c, centro in enumerate(km.cluster_centers_):
            fig.add_trace(go.Scatter(x=[centro[0]], y=[centro[1]], mode="markers+text",
                                     text=[mapa[c]], textposition="top center",
                                     marker=dict(symbol="x", size=12, color="black"),
                                     showlegend=False, hoverinfo="skip"))
    st.plotly_chart(fig)
    if entrada == ENTRADA_VAR:
        st.caption("K-Means se aplicó sobre las seis variables; el plano CP1–CP2 es solo una proyección "
                   "para dibujarlos, por lo que algunos países pueden verse cerca de otro grupo.")

    tam = grupos.value_counts().reindex(orden).reset_index()
    tam.columns = ["Grupo", "Países"]
    fig_t = px.bar(tam, x="Grupo", y="Países", text="Países", title="Tamaño de cada grupo")
    st.plotly_chart(fig_t)

    st.subheader("4.3 ¿Qué tan bien separados están?")
    ds = pd.DataFrame({"País": df["pais"], "Grupo": grupos, "Silueta": s})
    fig_s = px.box(ds, x="Grupo", y="Silueta", points="all", hover_name="País",
                   category_orders={"Grupo": orden}, title="Silueta por país, según el grupo asignado")
    fig_s.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_s)
    st.markdown("**Países más difíciles de asignar (menor silueta):**")
    st.dataframe(ds.sort_values("Silueta").head(10).round(3), hide_index=True)

    st.subheader("4.4 Perfil de los grupos con las variables originales")
    perfil = perfil_grupos(df, grupos, COLS_PERFIL)
    st.markdown("**Promedios por grupo (unidades originales):**")
    st.dataframe(perfil.rename(columns=ETIQUETAS).round(2))

    rel = perfil_relativo(df, grupos, COLS_PERFIL)
    fig_r = px.imshow(rel.round(2).values, x=[et(v) for v in COLS_PERFIL], y=list(rel.index),
                      color_continuous_scale="RdBu_r", color_continuous_midpoint=0, text_auto=True, aspect="auto",
                      title="Cada grupo frente al promedio global (en desviaciones estándar)")
    st.plotly_chart(fig_r)
    st.caption("Rojo: el grupo está por encima del promedio de los países; azul: por debajo. "
               "Las dos últimas columnas no se usaron para formar los grupos.")

    var = st.selectbox("Comparar una variable entre grupos", COLS_PERFIL, format_func=et)
    fig_v = px.box(df.assign(Grupo=grupos), x="Grupo", y=var, points="all", hover_name="pais",
                   category_orders={"Grupo": orden}, labels={var: et(var)},
                   title=f"{et(var)} por grupo")
    st.plotly_chart(fig_v)

    st.markdown("**Países de cada grupo:**")
    for tab, g in zip(st.tabs(orden), orden):
        with tab:
            st.write(", ".join(sorted(df.loc[grupos == g, "pais"])))

    st.subheader("4.5 Consultar un país")
    pais = st.selectbox("País", sorted(df["pais"]))
    i = int(df.index[df["pais"] == pais][0])
    st.markdown(f"**{pais}** pertenece al **{grupos.iloc[i]}** (silueta {s[i]:.2f}).")
    comp = pd.DataFrame({
        "Variable": [et(v) for v in COLS_PERFIL],
        pais: [df.loc[i, v] for v in COLS_PERFIL],
        f"Promedio {grupos.iloc[i]}": [perfil.loc[grupos.iloc[i], v] for v in COLS_PERFIL],
        "Promedio global": [df[v].mean() for v in COLS_PERFIL],
    })
    st.dataframe(comp.round(3), hide_index=True)

    st.download_button("Descargar la asignación de grupos (CSV)",
                       ds.to_csv(index=False).encode("utf-8"),
                       file_name=f"grupos_whr_{anio}.csv", mime="text/csv")


# ----------------------------------------------------------------------
# 5. Discusión
# ----------------------------------------------------------------------
LECTURA_REFERENCIA = """
**Configuración de referencia: año 2024, 2 componentes, K-Means sobre componentes, K = 3.**

**¿Qué variables definen los componentes?** El **CP1 (48,9 %)** reúne PIB per cápita (0,88),
esperanza de vida saludable (0,87) y apoyo social (0,81), con libertad (0,63) y percepción de
corrupción (0,60) en segundo plano. Lo leo como un eje de *condiciones materiales e
institucionales de vida*. El **CP2 (19,3 %)** casi se reduce a la **generosidad** (0,90), con aportes
menores de percepción de corrupción (0,38) y libertad (0,27), y el PIB en sentido contrario
(−0,29). La generosidad es casi independiente del resto (su correlación con el PIB es −0,08), por
eso forma su propia dimensión: el desarrollo y la generosidad no se mueven juntos.

**¿Cuánta variación se conserva?** Dos componentes retienen el **68,2 %** de la varianza (el criterio
de Kaiser retiene dos; con tres se llega a cerca del 81 %). Los gráficos en el plano CP1–CP2 resumen
ese 68,2 % y se pierde el 31,8 % restante.

**¿Qué diferencias reales hay entre los grupos?** Con K = 3 aparecen tres perfiles (las letras
solo los identifican):
- **Grupo A (62 países):** PIB y apoyo social algo por encima del promedio, pero con la
  **generosidad más baja** (−0,76 desviaciones) y la contribución de corrupción por debajo del
  promedio (−0,43, es decir, más corrupción percibida).
- **Grupo B (46 países):** las contribuciones más bajas en PIB (−1,17), apoyo social (−1,08),
  esperanza de vida (−0,92) y libertad (−0,56), con generosidad algo por encima del promedio (+0,47).
- **Grupo C (36 países):** contribuciones altas en casi todo, sobre todo corrupción percibida
  (+1,13), esperanza de vida (+1,04) y PIB (+0,96). Incluye a los países nórdicos y de Europa
  occidental, pero también a Kirguistán, Uzbekistán, Nicaragua, Kosovo y Tailandia, cuyo lugar se
  explica más por generosidad, libertad y apoyo social que por PIB.

Un resultado que respalda la lectura: la evaluación de vida **no se usó** para formar los grupos y
aun así los separa (promedios de 6,73 en el C, 5,83 en el A y 4,35 en el B).

**¿Son grupos claramente distinguibles?** Solo en parte. La silueta promedio de K = 3 es **0,46**
(K = 2 da 0,43), una estructura moderada y no grupos nítidos. **Siete países** tienen silueta menor
a 0,10 (Chequia es negativa, además de Indonesia, Irán, Líbano, Camboya, Namibia y Marruecos): están
en la frontera entre grupos y su asignación es frágil.

**Relación con el problema.** Los perfiles no son simplemente «países mejores y peores». Los grupos A
y B se diferencian a la vez en el nivel de condiciones de vida y en generosidad, algo que un
ranking de un solo número no muestra.
"""


def pagina_discusion():
    st.header("5. Discusión de resultados")

    n_comp = int(cfg.get("n_comp", kaiser))
    entrada = cfg.get("entrada", ENTRADA_CP)
    X = matriz_entrada(n_comp, entrada)
    ev = evaluar_k_cache(X, KS)
    k, k_sug, _ = resolver_k(n_comp, entrada, ev)
    labels, _km = ajustar_kmeans(X, k)
    grupos = nombrar_grupos(labels)
    s = silueta_por_observacion(X, labels)

    st.subheader("Resumen de tu configuración actual")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Año", int(anio))
    c2.metric("Componentes", n_comp if entrada == ENTRADA_CP else "6 variables")
    c3.metric("Varianza conservada", f"{acp['razon'][:n_comp].sum() * 100:.1f} %"
              if entrada == ENTRADA_CP else "100 %")
    c4.metric("K y silueta", f"{k} · {s.mean():.2f}")
    st.caption("Estos números se actualizan con lo que elijas en las secciones 3 y 4. "
               f"Tamaños de grupo: {', '.join(f'{g}: {n}' for g, n in grupos.value_counts().sort_index().items())}.")

    es_ref = (int(anio) == 2024 and entrada == ENTRADA_CP and n_comp == 2 and k == 3)
    if es_ref:
        st.success("Estás en la configuración de referencia: la lectura de abajo corresponde a estos resultados.")
    else:
        st.warning("La lectura de abajo se escribió para la **configuración de referencia** (2024, 2 componentes, "
                   "K-Means sobre componentes, K = 3). Con otra configuración, revisa qué conclusiones se mantienen.")
    st.markdown(LECTURA_REFERENCIA)

    st.subheader("Limitaciones que hay que tener presentes")
    st.markdown(
        "- **Variables modeladas:** los seis factores son contribuciones estimadas por el modelo del "
        "informe, no los indicadores crudos. Los resultados describen cómo el modelo reparte el puntaje "
        "entre factores.\n"
        "- **Información descartada:** K-Means sobre dos componentes ignora el 31,8 % de la varianza, y el "
        "CP2 es casi solo generosidad, por lo que parte de la agrupación depende de una sola variable. "
        "Con tres componentes la mejor silueta baja a 0,38 y sobre las seis variables a 0,30: la "
        "estructura de grupos es sensible a la decisión.\n"
        "- **Pesos iguales:** al estandarizar, la generosidad pesa igual que el PIB, aunque su "
        "contribución al puntaje es pequeña. Un ACP sobre covarianzas daría otra lectura.\n"
        "- **Supuestos de K-Means:** favorece grupos compactos y de tamaño parecido; no hay una "
        "frontera natural entre países y la elección de K es un juicio.\n"
        "- **Cobertura y tiempo:** son los países con datos ese año (se excluyeron los que tenían "
        "factores faltantes), de una sola foto; no permiten inferir causalidad."
    )


# ----------------------------------------------------------------------
# Navegación
# ----------------------------------------------------------------------
PAGINAS = {
    "1. Contexto y datos": pagina_contexto,
    "2. Exploración": pagina_exploracion,
    "3. ACP": pagina_acp,
    "4. K-Means": pagina_kmeans,
    "5. Discusión": pagina_discusion,
}
PAGINAS[pagina]()
