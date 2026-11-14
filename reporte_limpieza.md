# Reporte de depuración

- 1. Carga: 1969 filas x 13 columnas (2011-2024).
- 2. Duplicados: 0 filas eliminadas (filas idénticas o país-año repetido).
- 3. Cobertura: 1097 filas no traen ninguno de los seis factores (solo puntaje y ranking; filas por año: {2011: 156, 2012: 156, 2014: 158, 2015: 157, 2016: 155, 2017: 156, 2018: 156, 2023: 3}) y se excluyen del análisis.
- 4. Faltantes: 4 filas con algún factor ausente se eliminaron: ['Oman 2024', 'State of Palestine 2024', 'State of Palestine 2022', 'Tajikistan 2024'].
- 5. Rangos: 0 filas con factores negativos y 0 con puntaje fuera de [0, 10].
- 6. Consistencia: el puntaje coincide con la suma de sus componentes (diferencia máxima 0.003). Filas con diferencia > 0.01: 0.
-    Consecuencia: 'evaluacion_vida' y 'distopia_residual' NO entran al modelo (serían combinación lineal de las demás); se usan como variables suplementarias.
- 7. Ceros: 30 filas tienen algún factor igual a 0 (país en el mínimo de la muestra ese año); se conservan como valores reales.
- 8. Redundancia: ninguna pareja de variables supera |r| = 0.9 en el año más reciente.
- 9. Atípicos (2024, |z| > 3.0): ['Venezuela (pib_pc_log, z=-3.4)', 'Afghanistan (apoyo_social, z=-3.8)', 'Benin (apoyo_social, z=-3.1)', 'Afghanistan (libertad, z=-4.2)', 'Indonesia (generosidad, z=3.5)', 'Myanmar (generosidad, z=3.5)', 'Denmark (percepcion_corrupcion, z=3.0)', 'Finland (percepcion_corrupcion, z=3.2)', 'Singapore (percepcion_corrupcion, z=3.3)']. No se recortan ni eliminan.
- 10. Resultado: 868 filas país-año, 6 variables de modelo. Países por año: {2019: 153, 2020: 149, 2021: 146, 2022: 136, 2023: 140, 2024: 144}.

## Estadísticas descriptivas (2024)

```
       pib_pc_log  apoyo_social  esperanza_vida_sana  libertad  generosidad  percepcion_corrupcion  evaluacion_vida  distopia_residual
count     144.000       144.000              144.000   144.000      144.000                144.000          144.000            144.000
mean        1.321         1.330                0.551     0.749        0.114                  0.146            5.581              1.369
std         0.395         0.353                0.220     0.178        0.060                  0.113            1.167              0.522
min         0.000         0.000                0.000     0.000        0.000                  0.000            1.364             -0.099
25%         1.038         1.129                0.413     0.659        0.067                  0.066            4.666              1.103
50%         1.359         1.434                0.550     0.767        0.111                  0.118            5.869              1.387
75%         1.654         1.596                0.715     0.880        0.152                  0.182            6.492              1.678
max         2.028         1.840                0.948     1.018        0.323                  0.522            7.736              2.701
```
