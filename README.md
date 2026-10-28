# Bike Sharing Demand Forecast

Proyecto de Machine Learning para pronosticar la demanda horaria de
bicicletas compartidas usando Kedro.

## 1. Problema de negocio

Un operador de bicicletas compartidas necesita anticipar los períodos
de mayor y menor demanda para planificar sus recursos operativos.

**Pregunta de negocio:**

¿Cómo puede el operador pronosticar los alquileres de la próxima hora,
utilizando el calendario, las condiciones meteorológicas y la demanda
observada recientemente?

**Objetivo analítico:** regresión para estimar el número de alquileres
por hora.

**Stakeholder:** gerente de operaciones.

La predicción puede ayudar a reconocer horas de alta demanda, pero no
calcula directamente el número óptimo de bicicletas por estación,
porque el dataset no contiene inventario ni ubicaciones por estación.

## 2. Dataset

Fuente: Kaggle — Bike Sharing Demand
https://www.kaggle.com/competitions/bike-sharing-demand/data

El dataset contiene registros de alquileres por hora, calendario y
condiciones meteorológicas. La variable objetivo es `count`.

No se utilizan `casual` ni `registered` como predictores porque su suma
es exactamente igual a `count`, lo que causaría fuga de información.

Los archivos de entrada deben ubicarse en `data/01_raw/`:

- `train.csv`: datos etiquetados utilizados para el entrenamiento y
  la evaluación temporal.
- `test.csv`: datos de competición sin la variable objetivo. Se conserva
  como archivo original, pero no se utiliza en la evaluación actual.

Los archivos del dataset no se incluyen necesariamente en Git. Deben
obtenerse de la fuente original y colocarse en la ruta indicada.

## 3. Metodología

El proyecto sigue un flujo reproducible en Kedro:

1. Carga y validación de datos.
2. Ingeniería de variables temporales y meteorológicas.
3. Creación de rezagos mediante timestamps exactos.
4. Entrenamiento de CatBoost con objetivo `log1p(count)`.
5. Evaluación frente a un baseline de la misma hora de la semana anterior.

Los rezagos utilizados son 1, 24 y 168 horas, además de promedios móviles
de 24 y 168 horas. Solo se utilizan observaciones correspondientes a
timestamps anteriores al registro que se desea pronosticar.

Las horas que no tienen el historial completo necesario se excluyen.

## 4. Ejecución

Requisitos: Python 3.13 y las dependencias de `requirements.txt`.

Instalación:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Ejecutar el pipeline completo:

```bash
kedro run
```

La configuración del entrenamiento está en
`conf/base/parameters.yml`, y las rutas de entrada y salida en
`conf/base/catalog.yml`.

## 5. Resultados de validación

En la validación temporal de octubre a diciembre de 2012, con 695
observaciones:

| Modelo | RMSLE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Baseline semanal | 0.3790 | 50.39 | 86.16 | 0.8215 |
| CatBoost con rezagos | 0.2371 | 29.90 | 47.87 | 0.9449 |

El modelo con rezagos mejoró las cuatro métricas frente al baseline
en este período. Estas cifras son resultados de validación y no deben
interpretarse como un score garantizado en el leaderboard de Kaggle.

## 6. Artefactos

Después de ejecutar `kedro run`, se generan:

- `data/02_intermediate/model_features.csv`
- `data/08_reporting/validation_metrics.csv`
- `data/08_reporting/validation_predictions.csv`
- `data/06_models/catboost_model.pkl`

Las visualizaciones exploratorias se guardan en
`data/08_reporting/figures/` cuando se ejecutan los bloques de
análisis correspondientes del notebook.

## 7. Pruebas y calidad de código

Ejecutar Ruff:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

Ejecutar pruebas:

```bash
python -m pytest tests -q -p no:cacheprovider
```

GitHub Actions ejecuta la validación de código, las pruebas y una
ejecución de prueba de Kedro con datos sintéticos para comprobar que
el pipeline puede ejecutarse sin depender de los datos de Kaggle.

## 8. Limitaciones

- Los datos corresponden a 2011–2012 y no representan necesariamente
  la demanda de un sistema actual.
- El dataset de entrenamiento de la competición tiene huecos temporales.
  Por ello, los rezagos se calculan usando timestamps exactos y se
  excluyen los registros sin historial suficiente.
- El modelo está diseñado para un pronóstico de una hora adelante,
  actualizado con las observaciones reales más recientes. No equivale
  a pronosticar meses enteros sin recibir datos nuevos.
- Las condiciones meteorológicas deben estar disponibles al momento
  de generar la predicción. En una implementación real, las condiciones
  futuras tendrían que provenir de un pronóstico meteorológico.
- El modelo predice alquileres, no demanda insatisfecha, disponibilidad
  por estación ni la asignación óptima de bicicletas.
- La evaluación temporal es limitada y el rendimiento puede cambiar
  entre períodos. Se necesitan más pruebas temporales antes de usar
  el modelo en producción.

## 9. Reproducibilidad

El código, los parámetros, las pruebas y el workflow de GitHub Actions
permiten reconstruir el experimento. Los resultados deben interpretarse
conforme a los períodos y limitaciones descritos en este documento.
