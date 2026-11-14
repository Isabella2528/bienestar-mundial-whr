# ¿Qué dimensiones y perfiles de bienestar distinguen a los países?

**ACP y K-Means sobre los factores del World Happiness Report 2025**

Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali

- **Autor:** _[tu nombre]_
- **Aplicación desplegada:** _[enlace de Streamlit Community Cloud]_
- **Repositorio:** _[enlace de este repositorio]_

---

## 1. Contexto y fuente

### Pregunta del estudio

El *World Happiness Report* explica la evaluación de vida de cada país con seis factores: PIB per cápita, apoyo social, esperanza de vida saludable, libertad para decidir, generosidad y percepción de corrupción. Este proyecto busca responder dos preguntas:

1. ¿Qué **dimensiones** resumen las diferencias entre países en esos factores?
2. ¿Existen **perfiles de países** con una combinación parecida de condiciones, que un ranking de un solo número no deja ver?

### Observación y datos

- **Cada observación** es un país en un año. El análisis principal usa un solo año a la vez (por defecto 2024, con 144 países), de modo que cada país aparece una sola vez.
- **Fuente:** *World Happiness Report 2025*, archivo *Data for Figure 2.1* (<https://worldhappiness.report/>). El informe se basa en la Gallup World Poll.
- **Cobertura:** 1.969 filas país-año de 2011 a 2024. Los seis factores solo existen de 2019 a 2024; antes hay únicamente puntaje y ranking.

### Variables

| Variable | Rol | Qué representa |
|---|---|---|
| `pib_pc_log` | Modelo | Contribución del PIB per cápita (en logaritmo) a la evaluación de vida |
| `apoyo_social` | Modelo | Contribución del apoyo social |
| `esperanza_vida_sana` | Modelo | Contribución de la esperanza de vida saludable |
| `libertad` | Modelo | Contribución de la libertad para tomar decisiones de vida |
| `generosidad` | Modelo | Contribución de la generosidad |
| `percepcion_corrupcion` | Modelo | Contribución de la percepción de corrupción (más alto = menos corrupción percibida) |
| `evaluacion_vida` | Suplementaria | Puntaje de evaluación de vida, de 0 a 10 (promedio de 3 años) |
| `distopia_residual` | Suplementaria | Parte del puntaje que los seis factores no explican |

Las variables suplementarias **no entran** al ACP ni a K-Means: se usan para interpretar y validar los resultados.

### Limitaciones de la fuente

- Los seis factores son **contribuciones estimadas por el modelo del informe**, no los indicadores crudos. Los resultados describen cómo el modelo reparte el puntaje entre factores.
- Los datos provienen de encuestas con muestras limitadas por país y no cubren todos los países.
- Es una foto por año: no permite hablar de causalidad.

---

## 2. Proceso de depuración

La base se depura **antes** de la aplicación con `src/limpieza.py`, que se ejecuta una sola vez. La app solo lee los resultados, así que no hace trabajo pesado en tiempo real. La bitácora completa queda en `data/processed/reporte_limpieza.md`.

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Carga | Lectura del Excel y renombrado de columnas | 1.969 filas × 13 columnas |
| Duplicados | Verificación de filas idénticas y de país-año repetido | 0 duplicados |
| Cobertura | Se excluyen las filas sin ninguno de los seis factores (2011-2018 y 3 filas de 2023) | 1.097 filas excluidas |
| Faltantes | Se eliminan las filas con algún factor ausente: Omán 2024, Tayikistán 2024, Palestina 2022 y 2024. No se imputa: son menos del 1 % y la distopía se calcula a partir de los factores | 4 filas eliminadas |
| Rangos | Factores ≥ 0 y puntaje entre 0 y 10 | 0 filas fuera de rango |
| Consistencia | El puntaje coincide con la suma de los seis factores y la distopía (diferencia máxima 0,003) | `evaluacion_vida` y `distopia_residual` no pueden entrar al modelo: serían combinación lineal de las demás |
| Ceros | 30 filas tienen algún factor igual a 0: es el mínimo de la muestra ese año, no un dato faltante | Se conservan |
| Redundancia | Ninguna pareja de variables supera \|r\| = 0,90 | Sin cambios |
| Atípicos | Se reportan los países con \|z\| > 3 (por ejemplo, Afganistán en libertad, Indonesia y Myanmar en generosidad) | No se recortan ni eliminan: son países reales |
| Estandarización | Z-score calculado dentro de cada año | `whr_estandarizado.csv` |

**Resultado final:** 868 filas país-año (144 países en 2024) y 6 variables de modelo.

Archivos que genera `limpieza.py` en `data/processed/`:

- `whr_limpio.csv`: unidades originales, para perfilar los grupos.
- `whr_estandarizado.csv`: z-scores por año, como referencia. La app vuelve a estandarizar con `StandardScaler` para mostrar el paso.
- `reporte_limpieza.md`: bitácora de las decisiones.

---

## 3. Justificación metodológica

**Por qué ACP.** Los factores están correlacionados entre sí: PIB, esperanza de vida y apoyo social tienden a moverse juntos. El ACP resume esa información compartida en pocas dimensiones que se pueden dibujar e interpretar. Se aplica sobre datos estandarizados porque las escalas difieren (el PIB llega cerca de 2 y la generosidad a 0,3), y así ninguna variable domina solo por su magnitud. El número de componentes se decide con tres criterios (autovalor mayor que 1, varianza acumulada y gráfico de sedimentación), no con una cifra fijada de antemano.

**Por qué K-Means.** Un ranking de un solo número oculta que países con puntajes parecidos pueden tener combinaciones distintas de factores. K-Means permite explorar si hay perfiles de países y describirlos con las variables originales.

**Cómo se eligen los grupos.** Se compara K de 2 a 8 con el método del codo y el coeficiente de silueta. K-Means puede aplicarse sobre los componentes seleccionados o sobre las seis variables estandarizadas, y la app permite comparar ambas opciones. Los grupos se identifican con letras (Grupo A, B, C) ordenadas por tamaño; la letra no implica jerarquía ni calidad.

**Por qué un año a la vez.** Mezclar años repetiría a cada país varias veces y rompería la independencia entre observaciones. La estandarización dentro de cada año compara a cada país con sus pares del mismo año.

---

## 4. Resultados principales

Configuración de referencia: año 2024, 2 componentes, K-Means sobre componentes, K = 3. La app permite cambiarla y revisar cómo cambian los resultados.

### ACP

| Componente | Varianza explicada | Variables que lo definen (carga) |
|---|---|---|
| CP1 | 48,9 % | PIB (0,88), esperanza de vida (0,87), apoyo social (0,81); libertad (0,63) y corrupción (0,60) en segundo plano |
| CP2 | 19,3 % | Generosidad (0,90); corrupción (0,38) y libertad (0,27) aportan menos; PIB (−0,29) en sentido contrario |

- Dos componentes conservan el **68,2 %** de la varianza (el criterio de Kaiser retiene dos; con tres se llega a cerca del 81 %).
- El CP1 se lee como un eje de **condiciones materiales e institucionales de vida**. El CP2 es casi solo **generosidad**, que es casi independiente del resto (correlación con el PIB de −0,08).

### K-Means

| Grupo | Países | Perfil (en desviaciones estándar frente al promedio) |
|---|---|---|
| A | 62 | PIB y apoyo social algo por encima del promedio; generosidad más baja (−0,76) y más corrupción percibida (−0,43) |
| B | 46 | Contribuciones más bajas en PIB (−1,17), apoyo social (−1,08), esperanza de vida (−0,92) y libertad (−0,56); generosidad algo por encima del promedio (+0,47) |
| C | 36 | Contribuciones altas en casi todo: corrupción percibida (+1,13), esperanza de vida (+1,04), PIB (+0,96). Incluye a países nórdicos y de Europa occidental, y también a Kirguistán, Uzbekistán, Nicaragua, Kosovo y Tailandia, cuyo lugar se explica más por generosidad, libertad y apoyo social que por PIB |

- La **evaluación de vida no se usó** para formar los grupos y aun así los separa: promedios de 6,73 (C), 5,83 (A) y 4,35 (B).
- La silueta promedio es **0,46** para K = 3 (0,43 para K = 2): una estructura **moderada**, no grupos nítidos. Siete países tienen silueta menor a 0,10 (Chequia es negativa; también Indonesia, Irán, Líbano, Camboya, Namibia y Marruecos) y están en la frontera entre grupos.

### Lectura

Los perfiles no son simplemente «países mejores y peores»: los grupos A y B se diferencian a la vez en el nivel de condiciones de vida y en generosidad, algo que un ranking de un solo número no muestra. La discusión completa está en la sección 5 de la aplicación.

### Limitaciones del análisis

- K-Means sobre dos componentes ignora el 31,8 % de la varianza, y como el CP2 es casi solo generosidad, parte de la agrupación depende de una sola variable. Con tres componentes la mejor silueta baja a 0,38, y sobre las seis variables a 0,30: la estructura de grupos es sensible a esta decisión.
- Al estandarizar, la generosidad pesa igual que el PIB aunque su contribución al puntaje es pequeña. Un ACP sobre covarianzas daría otra lectura.
- K-Means favorece grupos compactos y de tamaño parecido, y la elección de K es un juicio.
- Los resultados cubren los países con datos completos ese año y no permiten inferir causalidad.

---

## 5. Cómo ejecutar el proyecto

Requiere Python 3.10 o superior.

```bash
# 1. Clonar e instalar
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt

# 2. (Opcional) Regenerar la base depurada a partir del Excel original
python src/limpieza.py

# 3. Ejecutar la aplicación
streamlit run app.py
```

Los archivos de `data/processed/` ya están incluidos, así que el paso 2 solo es necesario para reproducir la depuración.

### Dependencias (`requirements.txt`)

`streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly` y `openpyxl` (necesario para leer el Excel en `limpieza.py`). La aplicación dibuja con Plotly; `matplotlib` y `seaborn` están disponibles para análisis complementarios.

### Despliegue

La aplicación se despliega en Streamlit Community Cloud conectando este repositorio y seleccionando `app.py` como archivo principal. Usa rutas relativas, por lo que no requiere configuración adicional.

---

## 6. Estructura del repositorio

```
.
├── app.py                    Aplicación Streamlit (5 secciones)
├── requirements.txt          Dependencias
├── src/
│   ├── limpieza.py           Depuración previa de la base (se ejecuta una vez)
│   ├── analisis.py           Funciones de ACP, K-Means y perfilamiento
│   └── __init__.py
└── data/
    ├── raw/                  Excel original del World Happiness Report 2025
    └── processed/            Base limpia, base estandarizada y reporte de depuración
```

## 7. Cómo se generan los resultados

Todas las tablas, métricas y gráficos de la aplicación se calculan en tiempo de ejecución a partir de `data/processed/whr_limpio.csv`:

- **Estandarización, ACP, codo, silueta y K-Means:** funciones de `src/analisis.py` (scikit-learn), con semilla fija (`random_state = 42`) y 20 inicializaciones para que los resultados sean reproducibles.
- **Perfiles de grupo:** medias por grupo en unidades originales y frente al promedio global.
- **Visualizaciones:** Plotly, en `app.py`.

La aplicación tiene cinco secciones: contexto y datos, exploración, ACP, K-Means y discusión.
