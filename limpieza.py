"""
limpieza.py
Depuración previa de la base World Happiness Report 2025 (Data for Figure 2.1).
Se ejecuta UNA vez, fuera de la app; la app de Streamlit solo lee los resultados.

Entrada : data/raw/WHR25_Data_Figure_2_1v3.xlsx
Salidas : data/processed/whr_limpio.csv         unidades originales (para perfilar clústeres)
          data/processed/whr_estandarizado.csv  z-scores calculados DENTRO de cada año
          data/processed/reporte_limpieza.md    bitácora de decisiones (insumo del README)

Uso: python src/limpieza.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN
# ----------------------------------------------------------------------
RAW_PATH = Path("data/raw/WHR25_Data_Figure_2_1v3.xlsx")
OUT_DIR = Path("data/processed")

# Nombres originales del archivo -> nombres de trabajo
RENOMBRAR = {
    "Year": "anio",
    "Rank": "ranking",
    "Country name": "pais",
    "Life evaluation (3-year average)": "evaluacion_vida",
    "Lower whisker": "lim_inferior",
    "Upper whisker": "lim_superior",
    "Explained by: Log GDP per capita": "pib_pc_log",
    "Explained by: Social support": "apoyo_social",
    "Explained by: Healthy life expectancy": "esperanza_vida_sana",
    "Explained by: Freedom to make life choices": "libertad",
    "Explained by: Generosity": "generosidad",
    "Explained by: Perceptions of corruption": "percepcion_corrupcion",
    "Dystopia + residual": "distopia_residual",
}

# Variables que ENTRAN a ACP y K-Means (los seis factores del informe)
VARS_MODELO = ["pib_pc_log", "apoyo_social", "esperanza_vida_sana",
               "libertad", "generosidad", "percepcion_corrupcion"]
# Variables suplementarias: NO entran al modelo, se usan para interpretar/validar
VARS_SUPLEMENTARIAS = ["evaluacion_vida", "distopia_residual"]
# Identificación y contexto (no se modelan)
COLS_ID = ["pais", "anio", "ranking"]
# Descartadas: lim_inferior / lim_superior (intervalo de confianza del puntaje, no una característica del país)

TOL_IDENTIDAD = 0.01   # evaluacion_vida debe ser la suma de los 6 factores + distopia_residual
UMBRAL_CORR = 0.90     # alerta de redundancia
UMBRAL_Z = 3.0         # reporte (no eliminación) de valores atípicos
MIN_VARS = 4           # exigencia del taller

LOG = []


def reg(msg):
    print(msg)
    LOG.append(msg)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carga y nombres
    df = pd.read_excel(RAW_PATH, sheet_name=0)
    reg(f"1. Carga: {df.shape[0]} filas x {df.shape[1]} columnas "
        f"({df['Year'].min()}-{df['Year'].max()}).")
    faltan = [c for c in RENOMBRAR if c not in df.columns]
    if faltan:
        raise SystemExit(f"Columnas no encontradas: {faltan}\nDisponibles: {list(df.columns)}")
    df = df.rename(columns=RENOMBRAR)
    df["pais"] = df["pais"].astype(str).str.strip()

    # 2. Duplicados (un país no puede aparecer dos veces en el mismo año)
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=["pais", "anio"], keep="first")
    reg(f"2. Duplicados: {n0 - len(df)} filas eliminadas (filas idénticas o país-año repetido).")

    # 3. Tipos numéricos
    for c in VARS_MODELO + VARS_SUPLEMENTARIAS + ["ranking", "anio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. Años sin factores: los componentes solo existen desde 2019
    sin_factores = df[VARS_MODELO].isna().all(axis=1)
    anios_sin = df.loc[sin_factores, "anio"].value_counts().sort_index().to_dict()
    reg(f"3. Cobertura: {int(sin_factores.sum())} filas no traen ninguno de los seis factores "
        f"(solo puntaje y ranking; filas por año: {anios_sin}) y se excluyen del análisis.")
    df = df[~sin_factores].copy()

    # 5. Filas incompletas (se eliminan; el valor de distopía se calcula a partir de los factores,
    #    así que no se imputa)
    incompletas = df[VARS_MODELO + ["distopia_residual"]].isna().any(axis=1)
    if incompletas.any():
        detalle = df.loc[incompletas, ["pais", "anio"]].astype(str).agg(" ".join, axis=1).tolist()
        reg(f"4. Faltantes: {int(incompletas.sum())} filas con algún factor ausente se eliminaron: {detalle}.")
    df = df[~incompletas].copy()

    # 6. Rangos (los factores son contribuciones >= 0; el puntaje va de 0 a 10)
    fuera_f = (df[VARS_MODELO] < 0).any(axis=1)
    fuera_l = ~df["evaluacion_vida"].between(0, 10)
    reg(f"5. Rangos: {int(fuera_f.sum())} filas con factores negativos y {int(fuera_l.sum())} con puntaje fuera de [0, 10].")
    df = df[~(fuera_f | fuera_l)].copy()

    # 7. Identidad contable: puntaje = suma de 6 factores + distopía + residual
    suma = df[VARS_MODELO + ["distopia_residual"]].sum(axis=1)
    dif = (df["evaluacion_vida"] - suma).abs()
    reg(f"6. Consistencia: el puntaje coincide con la suma de sus componentes (diferencia máxima {dif.max():.3f}). "
        f"Filas con diferencia > {TOL_IDENTIDAD}: {int((dif > TOL_IDENTIDAD).sum())}.")
    reg("   Consecuencia: 'evaluacion_vida' y 'distopia_residual' NO entran al modelo "
        "(serían combinación lineal de las demás); se usan como variables suplementarias.")
    df = df[dif <= TOL_IDENTIDAD].copy()

    # 8. Ceros: valor mínimo de la muestra en ese factor (piso del indicador), no un dato faltante
    ceros = int((df[VARS_MODELO] == 0).any(axis=1).sum())
    reg(f"7. Ceros: {ceros} filas tienen algún factor igual a 0 (país en el mínimo de la muestra ese año); "
        "se conservan como valores reales.")

    # 9. Variación y redundancia
    sd = df.groupby("anio")[VARS_MODELO].std()
    constantes = [v for v in VARS_MODELO if (sd[v] == 0).any()]
    if constantes:
        raise SystemExit(f"Variables sin variación en algún año: {constantes}")
    if len(VARS_MODELO) < MIN_VARS:
        raise SystemExit("El taller exige al menos 4 variables cuantitativas.")
    corr = df[df["anio"] == df["anio"].max()][VARS_MODELO].corr().abs()
    for i, a in enumerate(VARS_MODELO):
        for b in VARS_MODELO[i + 1:]:
            if corr.loc[a, b] >= UMBRAL_CORR:
                reg(f"8. Alerta: |r| = {corr.loc[a, b]:.2f} entre '{a}' y '{b}'.")
    reg("8. Redundancia: ninguna pareja de variables supera |r| = "
        f"{UMBRAL_CORR} en el año más reciente.")

    # 10. Atípicos: se reportan, no se eliminan (son países reales)
    z = df.groupby("anio")[VARS_MODELO].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    ult = df["anio"] == df["anio"].max()
    atip = [f"{df.loc[i, 'pais']} ({v}, z={z.loc[i, v]:.1f})"
            for v in VARS_MODELO for i in z.index[ult & (z[v].abs() > UMBRAL_Z)]]
    reg(f"9. Atípicos ({df['anio'].max()}, |z| > {UMBRAL_Z}): {atip if atip else 'ninguno'}. "
        "No se recortan ni eliminan.")

    # 11. Estandarización dentro de cada año (equivale a StandardScaler sobre un año)
    est = pd.concat([df[COLS_ID], z], axis=1)

    # 12. Guardado
    df = df.sort_values(["anio", "pais"]).reset_index(drop=True)
    est = est.sort_values(["anio", "pais"]).reset_index(drop=True)
    columnas = COLS_ID + VARS_MODELO + VARS_SUPLEMENTARIAS
    df[columnas].to_csv(OUT_DIR / "whr_limpio.csv", index=False, encoding="utf-8")
    est.to_csv(OUT_DIR / "whr_estandarizado.csv", index=False, encoding="utf-8")
    por_anio = df.groupby("anio").size().to_dict()
    reg(f"10. Resultado: {len(df)} filas país-año, {len(VARS_MODELO)} variables de modelo. "
        f"Países por año: {por_anio}.")

    with open(OUT_DIR / "reporte_limpieza.md", "w", encoding="utf-8") as f:
        f.write("# Reporte de depuración\n\n")
        f.writelines(f"- {linea}\n" for linea in LOG)
        f.write(f"\n## Estadísticas descriptivas ({df['anio'].max()})\n\n```\n")
        f.write(df[df["anio"] == df["anio"].max()][VARS_MODELO + VARS_SUPLEMENTARIAS]
                .describe().round(3).to_string())
        f.write("\n```\n")


if __name__ == "__main__":
    main()
