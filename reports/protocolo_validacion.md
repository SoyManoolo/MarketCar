# Protocolo de validación y selección de modelos de MarketCar

Versión del protocolo: `1.0.0`. Este documento fija de antemano las particiones, el tratamiento de duplicados por vehículo, las métricas de evaluación, el esquema de validación anidada y la regla de decisión antes de reentrenar y comparar modelos (tarea #4).

## 1. Huella de reproducibilidad de datos y partición base

El protocolo reutiliza los metadatos de reproducibilidad introducidos en `src/modeling.py`:

- **Archivo de origen**: `data/cars_details_merges.csv`
- **SHA-256 del CSV bruto (`file_sha256`)**: `66c56000d1543402f4845e4b4b877ea839e4ef625100fe309ca2eabcb050d456`
- **Versión de limpieza (`cleaning_version`)**: `2.0.0` (código SHA-256: `42c8a9346104bd652ecc44a9418eb8cb3d0a03ef4013d110b287798033d28c7e`)
- **SHA-256 de la matriz limpia `X` (`dataframe_sha256`)**: `8d466af5a8dabdea30a2ddcd72af77251c92ba356b978134e1e951987a3839de`
- **SHA-256 del objetivo `y` (`target_sha256`)**: `c057ad82bc7d9c8ed06c30890dae45db965f7e9c46f3edf9debfb048ee4ce523`
- **Población generalista (`dynx_totalvalue_x <= 5,000,000`)**: `37,476` anuncios (`29,980` entrenamiento / `7,496` test)
- **Semilla base (`random_state`)**: `42`
- **SHA-256 de la partición 80/20 (`partition_sha256`)**: `451065db7e7aba3beedda20c5fdac318acbd80652ea48a98f3ec01c003aa7925`
- **Variables de entrada**: `51` (`30` numéricas y `21` categóricas)

## 2. Auditoría de duplicados por vehículo y agrupación

| Concepto | Recuento |
| --- | --- |
| Filas generalistas evaluadas | 37476 |
| Filas exactamente duplicadas en el CSV bruto | 0 |
| Grupos de vehículo únicos (componentes conexas de IDs) | 37473 |
| Grupos con múltiples anuncios del mismo vehículo (`ucid`/`sid`) | 3 |
| Filas pertenecientes a grupos con múltiples anuncios | 6 |
| Grupos de vehículo compartidos entre train y test | 0 |
| Filas test con las mismas características de entrada `X` que train | 27 |

### Detalle por identificador de vehículo

| Identificador | No nulos | Valores únicos | Filas repetidas | Filas test presentes en train |
| --- | --- | --- | --- | --- |
| usedCarId | 37476 | 37476 | 0 | 0 |
| usedCarSkuId | 37476 | 37476 | 0 | 0 |
| used_carid | 37476 | 37476 | 0 | 0 |
| ucid | 37476 | 37473 | 3 | 0 |
| sid | 37476 | 37473 | 3 | 0 |
| vlink | 37476 | 37476 | 0 | 0 |

### Regla sobre duplicados

1. **Agrupación por vehículo (`build_vehicle_groups`)**: existen 3 pares de anuncios (6 filas: índices `[2423, 2619, 17070, 32093, 37567, 37670]`) que comparten `ucid` y `sid` (reanuncios del mismo vehículo en el mismo vendedor con ligeras variaciones de precio o +1 km). En la partición 80/20 (`random_state=42`) las 6 filas caen en entrenamiento (`0` fugas hacia test), pero un `KFold` estándar sin agrupar las separa en folds distintos. Por ello, todas las particiones de validación cruzada agrupan por `vehicle_group_id`.
2. **No deduplicación por coincidencia de características**: las 27 filas de test que coinciden en variables de entrada `X` con alguna fila de entrenamiento tienen `usedCarId`, `usedCarSkuId`, `ucid` y `vlink` distintos. Representan coches diferentes del mismo modelo, año, ciudad y kilometraje redondeado, por lo que **no se eliminan**.

## 3. Folds comunes de entrenamiento y validación anidada

### 3.1. Alcance del conjunto de test actual y necesidad de validación anidada

El conjunto `X_test` (`7.496` anuncios) ya se consultó para comparar siete familias de modelos en el diagnóstico inicial. Cambiar la semilla aleatoria sobre el mismo CSV no crea un conjunto de test virgen, porque las decisiones de preprocesamiento y familias candidatas ya utilizaron esa muestra.

- **Mientras no se incorporen anuncios nuevos independientes**:
  - La **selección de hiperparámetros, comparación de objetivos (`log1p` vs rupias)** y **elección de modelo** se realizan **exclusivamente sobre `X_train` (`29.980` filas)** usando los folds comunes agrupados y validación anidada.
  - `X_test` se mantiene congelado únicamente como **referencia histórica de regresión** frente a los artefactos `v1`, declarando explícitamente que no constituye un test independiente virgen.

### 3.2. Folds comunes de entrenamiento (comparación directa de candidatos)

- **Folds principales de selección (`5` folds agrupados por `vehicle_group_id`)**:
  - Semilla: `42`
  - Huella SHA-256 (`splits_sha256`): `2b52bb93f036f2a3eebe94ba4bad93e097607c23d9833d7505165e55b5ebc8dd`
- **Folds diagnósticos rápidos (`3` folds agrupados por `vehicle_group_id`)**:
  - Semilla: `42`
  - Huella SHA-256 (`splits_sha256`): `b78288c630b07c22fbfdee36d12af12e3fac953fa1e4a0a642921b72175c4aa0`

| Fold | Filas train | Filas val | Grupos vehículo val | Media precio val INR | Mediana precio val INR |
| --- | --- | --- | --- | --- | --- |
| 1 | 23984 | 5996 | 5995 | 720645.08 | 520000.0 |
| 2 | 23984 | 5996 | 5995 | 714665.74 | 525000.0 |
| 3 | 23984 | 5996 | 5995 | 709420.09 | 517271.0 |
| 4 | 23984 | 5996 | 5996 | 726020.39 | 516500.0 |
| 5 | 23984 | 5996 | 5996 | 725949.55 | 523243.5 |

### 3.3. Protocolo de validación anidada (`build_nested_cv_splits`)

Cuando no se dispone de datos externos nuevos, la estimación no sesgada del rendimiento de un procedimiento de ajuste se obtiene mediante validación anidada sobre `X_train`:

- **Bucle externo (`5` folds agrupados por `vehicle_group_id`)**: semilla `42`, huella SHA-256 `2b52bb93f036f2a3eebe94ba4bad93e097607c23d9833d7505165e55b5ebc8dd`. Mide el error fuera de muestra de cada familia/procedimiento sin intervenir en el ajuste.
- **Bucle interno (`3` folds agrupados por `vehicle_group_id`)**: dentro de cada fold externo, selecciona hiperparámetros y compara transformación del objetivo (`log1p` frente a rupias directas) minimizando MAE en rupias.
- **Huellas SHA-256 de los bucles internos por fold externo**:
  - Fold externo 1: `ae11eb485e63333e1265bdcb16b24ef7681bff3c15d9166fc285f2aa24183bca`
  - Fold externo 2: `055d61475ca697d72211aecec43a01edd8770282c35363d64d8d867dce45be32`
  - Fold externo 3: `716043936f15c0ebbac65910389803eb273a3af55f2f5aec63e93d696a8278fe`
  - Fold externo 4: `523c18cc4f7eb7325d4b5e2ad78ffcb417cfd91d98ed3032253bc86a49ac0ff7`
  - Fold externo 5: `2d8bfadfa7e782a446202e0c9a34ae219e8fadc402efe2058216c19631c0d3a7`

## 4. Métricas de evaluación

Todas las predicciones se evalúan en la escala monetaria real (**rupias indias, INR**), aplicando `expm1` cuando el modelo entrene sobre `log1p(precio)`:

1. **Métrica principal**: **MAE (INR)** — error absoluto medio en rupias.
2. **Métricas complementarias globales**:
   - **RMSE (INR)** — raíz del error cuadrático medio (sensible a errores extremos).
   - **R²** — proporción de varianza explicada en rupias.
   - **Mediana APE (%)** — mediana de `|predicción - real| / real * 100`.
   - **Cobertura ±10 % (%)** — porcentaje de anuncios con error relativo `<= 10 %`.
   - **Sesgo medio (INR)** — media de `predicción - real` (negativo indica infravaloración).
3. **Desglose obligatorio por tramos de precio real (`evaluate_by_price_band`)**:

| Tramo de precio real (INR) | Segmento |
| --- | --- |
| (0, 300000] | Económico / entrada (≤ ₹3 Lakh) |
| (300000, 600000] | Compacto generalista (₹3–6 Lakh) |
| (600000, 1000000] | Medio (₹6–10 Lakh) |
| (1000000, 2000000] | Medio-alto / SUV (₹10–20 Lakh) |
| (2000000, 5000000] | Alto dentro del alcance (₹20–50 Lakh) |

## 5. Criterio de selección del modelo fijado de antemano

La función `select_best_candidate` en `src/validation.py` codifica la regla de decisión que se aplicará en la tarea #4 sobre las predicciones *out-of-fold* de validación:

1. **Ordenación primaria por MAE (INR)**: el candidato con menor MAE fuera de fold en validación es el líder inicial.
2. **Contraste de incertidumbre por bootstrap pareado (`paired_bootstrap_mae`)**: se calcula el intervalo de confianza al 95 % (`3,000` remuestreos pareados por anuncio, semilla `42`) de `MAE(líder) - MAE(candidato)`.
3. **Regla de empate técnico y parsimonia**: si el IC95 % incluye `0` y la diferencia absoluta de MAE es `<= ₹1,000`, los candidatos se consideran en empate técnico y se desempata en este orden estricto:
   1. Menor suma de dispersión entre folds (`SD` del MAE por fold) y **RMSE (INR)**.
   2. Menor **Mediana APE (%)** y mayor **cobertura ±10 % (%)**.
   3. Menor **MAE (INR)** en el tramo alto `(2000000, 5000000]`.
   4. Menor complejidad operativa (`complexity_rank`: Ridge < LightGBM/XGBoost/CatBoost < bosques masivos).
4. **Prohibición de selección sobre `X_test`**: ningún hiperparámetro ni elección entre familias u objetivos (`log1p` vs rupias) puede decidirse por su resultado en `X_test`.
