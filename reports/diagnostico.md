# Diagnóstico de MarketCar



Evaluación local de los siete artefactos existentes sobre la partición original (semilla 42, 80/20). Precios y errores monetarios en INR. No se modifican los modelos ni se realiza una nueva búsqueda de hiperparámetros.

## Reproducción de métricas

| Modelo | MAE INR | RMSE INR | R2 | MAE train guardado INR | Delta MAE vs guardado INR | Mediana APE % | Dentro ±10% % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM ajustado | 70137.447 | 143424.552 | 0.956 | 45724.059 | 0.0 | 7.46 | 61.993 |
| XGBoost ajustado | 70960.121 | 144832.934 | 0.955 | 52166.32 | 0.004 | 7.541 | 61.299 |
| CatBoost ajustado | 71321.13 | 147175.356 | 0.953 | 49180.409 | 0.0 | 7.566 | 61.086 |
| Extra Trees ajustado | 79255.933 | 163264.855 | 0.943 | 21276.001 | 0.0 | 8.628 | 55.696 |
| Random Forest ajustado | 80044.455 | 167063.324 | 0.94 | 39712.207 | 0.0 | 8.488 | 56.03 |
| Ridge Regression | 81326.494 | 160024.906 | 0.945 | 65400.56 | 0.0 | 9.087 | 53.829 |
| Linear Regression | 85497.495 | 171312.2 | 0.937 | 61201.504 | 0.0 | 9.363 | 52.922 |

El MAE de entrenamiento de esta tabla procede del artefacto guardado; las métricas de test se recalculan. Coincidir con las métricas guardadas respalda la reproducción, pero no certifica la identidad histórica de los datos.

## Independencia y calidad de datos

| Identificador | Filas test presentes en train |
| --- | --- |
| usedCarId | 0 |
| usedCarSkuId | 0 |
| used_carid | 0 |
| ucid | 0 |
| vlink | 0 |

Filas totalmente duplicadas en el CSV original: 0. Filas test con las mismas características de entrada que una fila train: 27 de 7496. Coincidencia de características no demuestra que sean el mismo vehículo; los identificadores no se usan como entradas.

Precio mínimo generalista: 11,963; precios no positivos: 0.

La limpieza común aplica reglas por fila. Imputación, escalado y one-hot se ajustan dentro de los pipelines y dentro de cada fold en los modelos sklearn. No se encontraron columnas de precio ni identificadores entre las entradas.

El corte de 5 millones se aplica al precio real antes de dividir. Define una población condicionada al objetivo; no proporciona un detector que decida si un coche nuevo está dentro del alcance. No equivale a excluir marcas premium.

## Problemas comprobados en la limpieza

| Problema | Filas |
| --- | --- |
| Length: comas convertidas a nulo | 344 |
| Width: comas convertidas a nulo | 344 |
| Height: comas convertidas a nulo | 346 |
| Wheel Base: comas convertidas a nulo | 635 |
| Max Torque: texto con unidad kg | 1364 |
| Max Power: texto con kW (a veces también bhp) | 19 |
| Kerb Weight: texto con comas | 107 |
| Kerb Weight: resultado inferior a 100 kg | 53 |

Recuentos sobre los 37.476 coches generalistas; las filas pueden aparecer en varias categorías. extract_first_number no normaliza unidades: el par en kgm queda mezclado con Nm. La potencia incluye algunos textos solo en kW y otros con bhp primero; el contador no significa que todos los 19 casos estén mal convertidos. Las dimensiones como `3,690 mm` se convierten en nulos; parse_weight divide `1,060 kg` en 1 y 60 y devuelve 30,5. Estas son anomalías del procesamiento, no evidencia de su impacto causal en el MAE. Corregir separadores y normalizar unidades con reglas verificadas, revisar valores físicamente improbables y volver a entrenar antes de comparar la mejora con validación de entrenamiento.

## Incertidumbre de la ventaja

| Comparación (LightGBM − otro) | Delta MAE INR | IC95 inferior INR | IC95 superior INR |
| --- | --- | --- | --- |
| xgboost | -822.674 | -1793.358 | 143.93 |
| catboost | -1183.682 | -2258.977 | -59.653 |
| ridge | -11189.046 | -13188.829 | -9094.215 |

Bootstrap pareado, 3.000 remuestreos de filas, semilla 42. Negativo favorece a LightGBM; si el intervalo incluye cero no hay una ventaja concluyente bajo este procedimiento. Supone filas independientes y no incorpora la variación del entrenamiento, la búsqueda ni la selección repetida sobre test.

## Validación cruzada común

| Modelo | Fold 1 INR | Fold 2 INR | Fold 3 INR | Media INR | SD folds INR |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 68718.945 | 68634.89 | 70749.711 | 69367.849 | 1197.466 |
| xgboost | 69969.008 | 69747.094 | 71495.875 | 70403.992 | 952.086 |

Tres folds barajados idénticos, únicamente sobre las 29.980 filas de entrenamiento; parámetros ganadores fijos. Estos datos ya intervinieron en la selección original: esta comparación es diagnóstica, no una evaluación anidada ni un test independiente. SD describe dispersión entre folds, no un intervalo de confianza.

## Metadatos de búsqueda

| Modelo | MAE CV original INR | Estrategia |
| --- | --- | --- |
| linear | — | LinearRegression + log1p(target) |
| ridge | 80617.117 | GridSearchCV(5 folds) + log1p(target) |
| random_forest | — | RandomForest ajustado + log1p(target) |
| extra_trees | 80538.366 | RandomizedSearchCV(24 candidatos, 3 folds) + log1p(target) |
| catboost | — | Parámetros seleccionados por RandomizedSearchCV previo |
| lightgbm | 69244.158 | RandomizedSearchCV(30 candidatos, 3 folds) + log1p(target) |
| xgboost | 70059.234 | RandomizedSearchCV(30 candidatos, 3 folds) + log1p(target) |

El módulo actual de CatBoost y su artefacto utilizan un CatBoostRegressor directo: sus predicciones están en rupias. El código de entrenamiento ajusta y_train sin log1p; los otros seis usan TransformedTargetRegressor con log1p/expm1. La comparación de métricas en rupias es válida, pero compara también objetivos de entrenamiento distintos. El historial de selección de CatBoost y Random Forest no es reproducible solo con los módulos actuales.

## LightGBM: errores por precio real

| Segmento | n | MAE INR | Mediana APE % | Sesgo INR |
| --- | --- | --- | --- | --- |
| (0, 300000] | 1772 | 31272.08 | 11.751 | 9080.676 |
| (300000, 600000] | 2564 | 43081.023 | 7.209 | -3121.153 |
| (600000, 1000000] | 1753 | 58406.804 | 5.654 | -7472.857 |
| (1000000, 2000000] | 1043 | 127856.956 | 6.461 | -13408.295 |
| (2000000, 5000000] | 364 | 341028.86 | 8.12 | -138003.49 |

Sesgo = predicción − precio real: negativo indica infravaloración.

## LightGBM: marcas con al menos 50 ejemplos test

| Segmento | n | MAE INR | Mediana APE % | Sesgo INR |
| --- | --- | --- | --- | --- |
| bmw | 146 | 309049.562 | 10.166 | -20962.609 |
| mercedes-benz | 171 | 275305.537 | 10.469 | -68906.329 |
| audi | 113 | 218601.185 | 8.164 | -43617.459 |
| mg | 87 | 127952.189 | 5.696 | -13971.898 |
| jeep | 78 | 124960.802 | 7.493 | 2096.865 |
| toyota | 261 | 101565.119 | 8.185 | -21421.316 |
| skoda | 204 | 97543.814 | 9.492 | -17056.963 |
| kia | 128 | 90077.517 | 4.78 | 4778.42 |
| mahindra | 406 | 81212.825 | 8.282 | 6199.499 |
| ford | 237 | 67494.22 | 9.628 | -10522.063 |
| tata | 405 | 61675.017 | 7.852 | 86.647 |
| volkswagen | 222 | 57465.249 | 7.918 | -15341.558 |
| nissan | 112 | 51629.968 | 9.95 | 2029.327 |
| honda | 781 | 51061.727 | 7.569 | -9320.306 |
| fiat | 56 | 46099.365 | 16.967 | -4633.377 |
| hyundai | 1490 | 44339.499 | 6.811 | -9275.208 |
| renault | 241 | 44315.925 | 7.114 | 2332.039 |
| maruti | 1987 | 36409.334 | 6.284 | -3692.558 |
| datsun | 86 | 34852.071 | 8.794 | -9291.149 |
| chevrolet | 173 | 32397.067 | 14.728 | -7154.774 |

## LightGBM: errores por año de fabricación

| Segmento | n | MAE INR | Mediana APE % | Sesgo INR |
| --- | --- | --- | --- | --- |
| (0, 2005] | 49 | 41692.853 | 19.605 | 6572.739 |
| (2005, 2010] | 783 | 42058.94 | 13.4 | -4461.879 |
| (2010, 2015] | 2697 | 65233.323 | 9.142 | -12656.259 |
| (2015, 2020] | 3209 | 78633.53 | 6.203 | -7285.716 |
| (2020, 2100] | 758 | 82461.656 | 4.754 | -11272.216 |

## Comparación por precio

| Precio real | n | MAE lightgbm INR | MAE xgboost INR | MAE catboost INR |
| --- | --- | --- | --- | --- |
| (0, 300000] | 1772 | 31272.08 | 31077.357 | 33485.236 |
| (300000, 600000] | 2564 | 43081.023 | 43481.45 | 43978.101 |
| (600000, 1000000] | 1753 | 58406.804 | 60303.806 | 61013.883 |
| (1000000, 2000000] | 1043 | 127856.956 | 128449.51 | 124342.265 |
| (2000000, 5000000] | 364 | 341028.86 | 345264.135 | 345827.42 |

## Cola de errores y cobertura

El 1% de filas con mayor error absoluto concentra el 54.13% del error cuadrático total. Mediana del error absoluto: 37,727; percentil 95: 239,118; percentil 99: 580,230. Sesgo medio: -9,236.

lightgbm: MAE sin coincidencias de características train/test (7469 filas) = 70,183.

xgboost: MAE sin coincidencias de características train/test (7469 filas) = 71,014.

catboost: MAE sin coincidencias de características train/test (7469 filas) = 71,404.

brand_name: 2 filas test con categoría no vista en train.

model_name: 7 filas test con categoría no vista en train.

variant_name: 276 filas test con categoría no vista en train.

Categorías conocidas: n=7220, MAE LightGBM=66,298.

Alguna categoría nueva: n=276, MAE LightGBM=170,569.

## Diez mayores errores de LightGBM

| brand_name | model_name | model_year | actual | prediction | absolute_error |
| --- | --- | --- | --- | --- | --- |
| land rover | land rover discovery 4 | 2013 | 4390000.0 | 1867060.049 | 2522939.951 |
| audi | audi a4 | 2021 | 4900000.0 | 2673657.334 | 2226342.666 |
| bmw | bmw 6 series | 2011 | 5000000.0 | 2806031.825 | 2193968.175 |
| land rover | land rover range rover sport | 2011 | 3700000.0 | 1627505.8 | 2072494.2 |
| bmw | bmw x1 | 2019 | 5000000.0 | 2979580.362 | 2020419.638 |
| bmw | bmw 7 series | 2017 | 2699000.0 | 4550686.796 | 1851686.796 |
| hyundai | hyundai tucson | 2022 | 3600000.0 | 2125447.46 | 1474552.54 |
| mercedes-benz | mercedes-benz m-class | 2014 | 3200000.0 | 1789426.646 | 1410573.354 |
| bmw | bmw x3 | 2011 | 2100000.0 | 742244.731 | 1357755.269 |
| mercedes-benz | mercedes-benz gle | 2018 | 5000000.0 | 3643870.182 | 1356129.818 |

## Riesgos de reproducibilidad y próximos pasos

1. Los artefactos no guardan huella del dataset, índices de partición, versión del preprocesamiento ni versiones de librerías. La comprobación de carga solo valida nombres de variables, objetivo y límite; puede aceptar un modelo entrenado con otra revisión de datos. Guardar estas huellas en futuros entrenamientos.

2. El test se consulta para comparar siete modelos y ya ha guiado decisiones. Reservar datos nuevos independientes o utilizar validación anidada antes de afirmar el rendimiento final; cambiar la semilla sobre datos ya usados no crea un test virgen para los modelos existentes.

3. Revisar los anuncios con mayores errores antes de excluirlos; distinguir fallos de datos, poca representación y variables relevantes ausentes.

4. Comparar un objetivo en rupias con log1p usando solo validación de entrenamiento, y calibrar intervalos en datos reservados. El logaritmo puede favorecer errores relativos, pero el sesgo observado no permite atribuirle causalidad por sí solo.

5. Validar temporalmente cuando exista fecha de anuncio y definir una política de alcance observable antes de conocer el precio real.
