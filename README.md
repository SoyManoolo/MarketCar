# MarketCar

Proyecto de aprendizaje de *machine learning* para estimar el precio de coches de
segunda mano. La primera versión usa datos de coches usados de India y predice la
columna `dynx_totalvalue_x`, expresada en rupias indias (₹).

> El modelo actual es un prototipo técnico. No debe usarse para tasar vehículos
> españoles: para ello habrá que entrenar de nuevo con datos del mercado español.

## Objetivo

Construir un flujo reproducible que, a partir de características como marca,
modelo, versión, año, kilometraje, combustible, potencia, ubicación y equipamiento,
estime un precio de mercado.

El proyecto ya compara tres familias de modelos:

1. **Linear Regression**, como línea base sencilla.
2. **Ridge Regression**, una regresión lineal regularizada.
3. **Random Forest Regressor**, un modelo de árboles no lineal.

## Modelos evaluados y pendientes

Los próximos experimentos conservarán la misma partición de entrenamiento y test,
la transformación `log1p(precio)` y las mismas métricas. Así la comparación será
justa frente a Ridge y Random Forest.

| Estado | Modelo | Motivo para probarlo |
|---:|---|---|
| Evaluado | CatBoost Regressor | Maneja directamente `brand_name`, `model_name` y `variant_name`, sin crear miles de columnas one-hot, y tolera nulos de forma nativa. |
| Evaluado | LightGBM Regressor | Gradient boosting rápido y eficaz. Es el mejor modelo actual en las tres métricas de test. |
| Pendiente | XGBoost Regressor | Modelo de gradient boosting muy consolidado y potente; permite una comparación sólida con LightGBM y CatBoost. |
| Pendiente | ExtraTreesRegressor | Variante de árboles aleatorizados relacionada con Random Forest. Es sencillo de comparar con el mismo preprocesamiento y puede generalizar mejor en algunos casos. |

También se probará, cuando se seleccione un modelo principal, una versión con
**regresión por cuantiles**. El objetivo no es sustituir la predicción puntual,
sino devolver un rango de tasación, por ejemplo: “precio estimado: ₹900.000;
rango probable: ₹820.000–₹1.000.000”. CatBoost, LightGBM y XGBoost permiten este
tipo de objetivo.

No se priorizan KNN ni SVR. KNN y SVR suelen comportarse peor o
ser más lentos con decenas de miles de filas y muchas categorías one-hot. 

Probar nuevos algoritmos no elimina el límite de los vehículos premium raros: si
hay pocos Ferrari, AMG o BMW Serie 7 en entrenamiento, ningún modelo tendrá datos
suficientes para aprender una tasación fiable de esos casos. Para ellos puede ser
necesario un segmento o modelo especializado, siempre que se recopilen suficientes
ejemplos.

## Datos y variable objetivo

Los datos no se incluyen en este repositorio. Proceden del dataset
[Used Cars Dataset (CarDekho)](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho)
de Sukrit Chatterjee en Kaggle, publicado bajo licencia **CC BY-NC 4.0**. Es un
recurso destinado a fines académicos y de investigación, por lo que no debe
utilizarse con fines comerciales.

Para reproducir el experimento, descarga el dataset desde Kaggle y coloca sus
archivos CSV en la carpeta `data/`. El flujo actual utiliza
`data/cars_details_merges.csv`; `feature_dictionary.csv` sirve como referencia de
las columnas. Los CSV se excluyen de Git deliberadamente para respetar la licencia
del dataset y evitar versionar archivos pesados.

Después de los filtros de calidad iniciales se conservan **37.809 coches**. El
tasador generalista limita su alcance a precios de hasta ₹5.000.000: conserva
**37.476 coches** y separa **333 vehículos premium** (el 0,88 %) para un posible
modelo especializado futuro. Se divide el conjunto generalista de forma
reproducible en:

| Conjunto | Proporción | Uso |
|---|---:|---|
| Entrenamiento | 80 % (29.980 coches) | Ajustar modelos y seleccionar parámetros. |
| Test | 20 % (7.496 coches) | Medir el comportamiento sobre coches no vistos. |

La semilla `random_state=42` permite repetir la misma partición.

## Limpieza y creación de variables

La función `clean_data` en `src/data_cleaning.py` realiza la preparación común
antes de crear `X` e `y`.

- Elimina kilometrajes imposibles, superiores a 1.000.000 km.
- Elimina precios superiores a ₹100.000.000, incluido un anuncio anómalo de una
  Ford EcoSport de 2016 con un valor de ₹550.000.555.
- Excluye variables con demasiados nulos: `Gross Weight`, `Compression Ratio` y
  `Ground Clearance Unladen`.
- Normaliza texto categórico a minúsculas, sin espacios sobrantes, y representa
  valores ausentes como `unknown`.
- Convierte medidas guardadas como texto —dimensiones, peso, potencia, par,
  velocidad, aceleración, volumen y llantas— a números.
- Convierte las listas de equipamiento en contadores, por ejemplo
  `safety_features_count`.
- Separa `mileage_new` en su valor numérico y `mileage_unit` (`kmpl`, `km/kg` o
  `unknown`).
- Convierte turbo y supercharger a valores binarios y añade indicadores de ausencia
  (`Turbo Charger_missing` y `Super Charger_missing`).

### Casos mecánicos especiales

`Displacement` es la cilindrada del motor térmico en cc. El valor cero no se trata
como nulo para coches eléctricos: en el conjunto hay 104 filas con
`Displacement = 0` y todas corresponden a `fuel_type = electric`. Un vehículo
eléctrico no tiene cilindrada, por lo que conservar el cero es correcto.

En cambio, para un vehículo no eléctrico, una cilindrada cero se considera un dato
inválido y se transforma a `NA`.

La columna `Values per Cylinder` tiene un nombre confuso: el diccionario de datos
la describe como el número de **válvulas por cilindro**, no como el número de
cilindros. En coches eléctricos ese concepto no aplica, por lo que se transforma a
`NA`. Los valores superiores a 12 en vehículos térmicos también se consideran
imposibles y se transforman a `NA`.

## Preprocesamiento y prevención de fuga de información

Después de separar entrenamiento y test, los modelos reciben los datos a través de
`Pipeline` y `ColumnTransformer`.

### Variables numéricas

1. Los nulos se imputan con la **mediana calculada solo sobre entrenamiento**.
2. Se crean indicadores de ausencia: el modelo puede distinguir entre un valor
   observado y otro que fue imputado.
3. Linear Regression y Ridge escalan los valores con `StandardScaler`. Random
   Forest no usa escalado porque sus divisiones se basan en umbrales, no en
   distancias ni en coeficientes lineales.

Imputar no afirma que el valor real sea la mediana: evita descartar un coche con
información útil en otras columnas. El indicador de ausencia conserva la señal de
que el dato no estaba disponible.

### Variables categóricas

Las categorías se convierten a columnas binarias mediante `OneHotEncoder`. Por
ejemplo, los valores de combustible generan columnas como `fuel_type_petrol`,
`fuel_type_diesel` y `fuel_type_electric`. La opción `handle_unknown="ignore"`
evita errores si aparece en test una categoría que no existía en entrenamiento.

## Transformación del precio

Todos los modelos se entrenan con `log1p(precio)` y sus predicciones se devuelven a
rupias con `expm1`. Esto reduce el peso de los precios muy altos durante el
entrenamiento y hace la optimización más estable. Las métricas finales, sin embargo,
se calculan siempre en rupias reales.

## Métricas

Se usan tres métricas complementarias:

| Métrica | Interpretación | Objetivo |
|---|---|---|
| MAE | Error absoluto medio en rupias. | Cuanto menor, mejor. |
| RMSE | Error que penaliza mucho más los fallos grandes. | Cuanto menor, mejor. |
| R² | Proporción de variación del precio explicada por el modelo. | Cuanto mayor, mejor. |

También se calculó una mediana de error absoluto de **₹45.782** para la regresión
lineal: aproximadamente la mitad de los coches de test tuvo un error inferior a esa
cifra. Esta métrica es menos sensible a unos pocos coches extremadamente caros.

Las métricas de entrenamiento son diagnósticas; las métricas de test son las que
simulan el uso con coches nuevos.

## Resultados obtenidos antes del límite de ₹5 millones

Los resultados siguientes se obtuvieron antes de limitar el modelo al mercado
generalista. Se mantienen como referencia del análisis de vehículos premium y se
usan para cuantificar el efecto del cambio de alcance.

| Modelo | MAE train | MAE test | RMSE train | RMSE test | R² train | R² test |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | ₹64.978 | ₹99.335 | ₹136.105 | ₹352.489 | 0,986 | 0,880 |
| Ridge (`alpha=1`) | ₹72.242 | ₹95.078 | ₹209.322 | ₹330.910 | 0,967 | **0,894** |
| Random Forest ajustado | ₹49.150 | **₹93.544** | ₹459.305 | ₹428.917 | 0,843 | 0,822 |

### Interpretación antes del límite

La regresión lineal logra un ajuste muy alto en entrenamiento, pero empeora al pasar
a test. Esto indica sobreajuste moderado, favorecido por las muchas categorías muy
específicas.

Ridge elige `alpha=1` mediante `GridSearchCV` de 5 folds y reduce el salto entre
train y test. Aunque su MAE no es el menor, es el modelo más equilibrado: tiene el
menor RMSE y el mejor R² en test. Por ello fue el **baseline ganador en el
conjunto sin límite de precio**.

Random Forest reduce el MAE de test en ₹1.534 respecto a Ridge, pero la mejora es
pequeña (aproximadamente un 1,6 %) y empeora mucho RMSE y R². Es más preciso para
muchos coches habituales, pero tiene fallos absolutos muy grandes en casos raros.

## Resultados del modelo generalista (hasta ₹5 millones)

Al excluir del alcance los vehículos con precio superior a ₹5.000.000, las métricas
se vuelven a calcular desde cero con una nueva división reproducible de los 37.476
coches restantes.

| Modelo | MAE train | MAE test | RMSE train | RMSE test | R² train | R² test |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | ₹61.202 | ₹85.497 | ₹111.528 | ₹171.312 | 0,973 | 0,937 |
| Ridge (`alpha=1`) | ₹65.401 | ₹81.326 | ₹117.507 | **₹160.025** | 0,970 | **0,945** |
| Random Forest ajustado | ₹39.712 | **₹80.044** | ₹83.457 | ₹167.063 | 0,985 | 0,940 |
| CatBoost ajustado | ₹49.180 | **₹71.321** | ₹75.224 | **₹147.175** | 0,988 | **0,953** |
| LightGBM ajustado | ₹45.724 | **₹70.137** | ₹77.822 | **₹143.425** | 0,987 | **0,956** |

### Efecto cuantificado de limitar el alcance

El cambio no hace que los algoritmos aprendan mejor por sí mismo: cambia la
población evaluada y elimina el 0,88 % de vehículos premium con precios extremos y
muy poca representación. Dentro del alcance definido, la mejora de las métricas de
test es sustancial:

| Modelo | Cambio en MAE test | Cambio en RMSE test | Cambio en R² test |
|---|---:|---:|---:|
| Linear Regression | −₹13.838 (−13,9 %) | −₹181.177 (−51,4 %) | +0,057 |
| Ridge | −₹13.752 (−14,5 %) | −₹170.885 (−51,6 %) | +0,051 |
| Random Forest | −₹13.500 (−14,4 %) | −₹261.854 (−61,0 %) | +0,118 |

La mayor reducción relativa de RMSE confirma que los vehículos muy caros estaban
dominando los errores extremos, en especial para Random Forest. Esto es una mejora
válida para un tasador de mercado convencional, pero no demuestra significación
estadística formal ni autoriza a usar el modelo con vehículos premium. Para afirmar
significación estadística haría falta calcular intervalos de confianza, por ejemplo
con *bootstrap* de los errores de test.

Dentro del alcance generalista, LightGBM es el mejor modelo actual: obtiene el
menor MAE, el menor RMSE y el mayor R² en test. Frente a CatBoost, reduce el MAE
en ₹1.184, el RMSE en ₹3.750 y eleva R² de 0,953 a 0,956. La mejora es pequeña,
pero consistente en las tres métricas.

### Configuración ganadora de LightGBM

La búsqueda aleatoria de 30 configuraciones con validación cruzada de tres folds
seleccionó los siguientes parámetros:

```python
LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=0,
)
```

El MAE medio de validación cruzada fue **₹69.244**, frente a **₹70.137** en el
conjunto de test: una diferencia de solo ₹893. Esta cercanía es una señal favorable
de que los hiperparámetros elegidos sobre entrenamiento se comportan de manera
similar en datos no vistos.

El modelo entrenado se guarda en `models/lightgbm_generalista.joblib`, junto con
los parámetros seleccionados, las métricas y la lista de variables esperadas.

### Configuración de CatBoost

CatBoost se entrenó una sola vez con la configuración seleccionada previamente por
validación cruzada:

```python
CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_strength=0.5,
    bagging_temperature=2,
)
```

## Ajuste de Random Forest

Primero se probaron combinaciones de `max_depth`, `min_samples_leaf` y
`max_features` con validación cruzada de 3 folds. Después se refinó la búsqueda con
profundidades 20 y 30, y 100, 200 y 400 árboles.

La configuración seleccionada por MAE fue:

```python
RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=42,
)
```

`n_estimators` aporta estabilidad al promedio del bosque, pero no es el principal
control de sobreajuste. Los parámetros más relevantes son:

- `max_depth`: limita la complejidad de cada árbol.
- `min_samples_leaf`: evita hojas que representen solo uno o pocos anuncios.
- `max_features`: introduce diversidad entre árboles usando una fracción de las
  variables en cada división.

## Errores por tramo de precio antes del límite

| Rango de precio real | MAE Ridge | MAE Random Forest |
|---|---:|---:|
| ₹25.000–₹276.887 | **₹32.238** | ₹33.839 |
| ₹276.887–₹435.000 | ₹42.627 | **₹42.570** |
| ₹435.000–₹625.000 | ₹55.508 | **₹54.399** |
| ₹625.000–₹980.000 | ₹75.976 | **₹72.529** |
| ₹980.000–₹32.570.569 | ₹269.418 | **₹264.722** |

Random Forest gana por poco en cuatro de los cinco tramos, pero el último agrupa
desde coches relativamente caros hasta vehículos exóticos. Por eso el promedio de
errores absolutos por tramo no refleja por sí solo los peores casos.

### Vehículos premium y errores extremos

Los diez mayores errores de Random Forest corresponden a vehículos premium muy poco
frecuentes, como Ferrari 488, Toyota Land Cruiser, BMW Serie 7 y Mercedes-AMG.

El caso más extremo es un Ferrari 488 GTB Spider de 2020 con 5.000 km:

| Valor | Precio |
|---|---:|
| Precio real | ₹32.570.569 |
| Predicción Ridge | ₹14.624.360 |
| Predicción Random Forest | ₹2.838.429 |
| Error absoluto de Random Forest | ₹29.732.140 |

Ese único caso aporta aproximadamente ₹342.000 al RMSE de Random Forest. Los árboles
predicen medias de las observaciones que llegan a cada hoja y no extrapolan bien
hacia precios extremos. Además, al entrenar sobre el logaritmo del precio, el modelo
prioriza el error relativo y puede infravalorar en rupias vehículos muy caros.

Estas filas no se deben eliminar automáticamente: representan coches reales. La
limitación es que hay pocos ejemplos similares para aprender una tasación fiable.

## Limitaciones actuales

- El precio objetivo es precio anunciado, no necesariamente precio final de venta.
- No se conocen variables importantes como estado real, accidentes, mantenimiento,
  garantía o historial completo.
- Las versiones y marcas raras tienen pocos ejemplos.
- El test es una partición aleatoria; cuando haya fecha de anuncio, convendrá probar
  también una división temporal.
- El modelo de India no se transfiere directamente al mercado español.

## Próximos pasos

1. Probar **XGBoost Regressor**, manteniendo la misma partición para compararlo de
   forma justa con CatBoost y LightGBM.
2. Revisar errores por marca, precio y antigüedad; separar coches convencionales de
   lujo si se dispone de suficientes ejemplos premium.
3. Generar rangos de tasación, no solo un precio puntual, por ejemplo mediante
   modelos de cuantiles o predicción conformal.
4. Recopilar datos españoles con permisos y condiciones de uso adecuados, normalizar
   sus campos y reentrenar el flujo completo.
5. Validar con particiones temporales y mantener un conjunto final sin usar durante
   el ajuste de hiperparámetros.

## Ejecución y modelos guardados

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe .\src\main.py --models lightgbm
```

Cada modelo tiene su propio módulo en `src/models/` y `main.py` actúa como
orquestador. Puedes elegir uno o varios modelos:

```powershell
# Ejecutar o cargar un único modelo.
.\.venv\Scripts\python.exe .\src\main.py --models catboost

# Ejecutar o cargar varios modelos.
.\.venv\Scripts\python.exe .\src\main.py --models ridge random_forest lightgbm

# Ejecutar todos los modelos disponibles.
.\.venv\Scripts\python.exe .\src\main.py --models all
```

Después del primer entrenamiento, cada modelo se guarda como un artefacto `joblib`
en `models/`, por ejemplo `models/lightgbm_generalista.joblib`. El artefacto incluye
el modelo entrenado, su pipeline de preprocesamiento, la lista de variables, el
límite de precio y las métricas obtenidas.

En ejecuciones posteriores, `main.py` carga el archivo existente y evalúa el modelo
sin reentrenarlo. Para forzar un entrenamiento nuevo, por ejemplo tras modificar
los datos o los hiperparámetros, usa:

```powershell
.\.venv\Scripts\python.exe .\src\main.py --models lightgbm --retrain
```

Los archivos `models/*.joblib` son binarios locales y están excluidos de Git. No se
deben cargar artefactos `joblib` procedentes de fuentes no fiables.
