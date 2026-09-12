# Predicción de categorías de compra a partir de títulos de licitaciones públicas chilenas

[![CI](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-83%20passing-brightgreen)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![data](https://img.shields.io/badge/data-CC0%20ChileCompra-lightgrey)](https://api.mercadopublico.cl)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Español**

Chile publica todas sus licitaciones públicas a través de una API OCDS abierta, y
cada una llega con dos cosas: el título en texto libre que escribió un
funcionario, y la categoría UNSPSC oficial bajo la que quedaron registrados sus
ítems. Ese par es un dataset etiquetado producido como subproducto de la
administración del Estado.

Este proyecto pregunta si ocho palabras de español abreviado y en mayúsculas
bastan para recuperar la categoría, y si un LLM local lo hace mejor que un modelo
lineal.

**En zero-shot, no lo hace**: el LLM pierde por 27,8 puntos de accuracy y tarda
624 veces más. Con ocho ejemplos *recuperados* cierra buena parte de esa
brecha — pero también lo hace un simple voto mayoritario sobre esos mismos ocho
ejemplos, sin ningún modelo de lenguaje de por medio, a un quinceavo de la
latencia. Ocho ejemplos *aleatorios* no cambian nada. Los dos brazos de control
existen porque sin ellos la lectura favorable es la única disponible.

---

## Resultados

Conjunto de prueba: abril de 2026 (952 licitaciones). Entrenado con enero a marzo
(2.874). 32 categorías tras agrupar la cola de clases raras.

| Modelo | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| Clase mayoritaria | 0,1429 | [0,1208; 0,1670] | 0,0078 | 0,00 |
| Reglas por palabra clave | 0,2384 | [0,2111; 0,2658] | 0,1295 | 0,08 |
| **TF-IDF + SVM lineal** | **0,5578** | **[0,5263; 0,5893]** | **0,5077** | **0,26** |
| LLM local (qwen2.5:7b) | 0,2794 | [0,2511; 0,3088] | 0,2491 | 162,19 |

Bootstrap apareado sobre 5.000 remuestreos, cada modelo sobre las 952:

| Comparación | Δ accuracy | IC 95 % | Significativo |
|---|---:|:--|:--:|
| TF-IDF vs. palabras clave | +0,3193 | [+0,2805; +0,3571] | sí |
| **LLM vs. TF-IDF** | **−0,2784** | **[−0,3172; −0,2395]** | **sí** |

El modelo supervisado aprende el vocabulario de las compras públicas chilenas a
partir de 2.874 ejemplos. En zero-shot, el LLM tiene que razonarlo desde un
prompt, y ocho palabras abreviadas en mayúsculas no dan mucho con qué razonar.

> **Corrección.** Esta tabla reportaba el LLM a **2.242,89 ms** por licitación y
> **9.345×** el costo del modelo lineal. Unos 2.054 ms de eso eran el cliente y
> no el modelo: `localhost` resuelve primero a `::1`, Ollama escucha solo en
> IPv4, y `urllib` espera a que Windows rechace la conexión IPv6 antes de caer a
> la otra dirección — en cada llamada, en serie.
> [`scripts/measure_llm_overhead.py`](scripts/measure_llm_overhead.py) cronometra
> un endpoint que no computa nada y encuentra 2.055,4 ms por el nombre contra
> 1,3 ms por la IP.
>
> También cambió qué era costeable. A 2,2 s por licitación una pasada completa
> tomaba 36 minutos, así que el LLM se puntuaba sobre un subconjunto de 300 y la
> comparación venía con una advertencia sobre filas pareadas. A 162 ms toma dos
> minutos y medio, así que cada fila de arriba es el conjunto de prueba completo
> y la advertencia desapareció — y por eso la brecha aquí es de 27,8 puntos y no
> de los 32 publicados antes.

### Después se le dieron ejemplos

Mismo mes de prueba, mismo subconjunto de 200 licitaciones con seed fijo, k = 8:

| Variante | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| TF-IDF + SVM lineal | 0,5100 | [0,4400; 0,5800] | 0,4400 | 0,28 |
| LLM zero-shot | 0,3150 | [0,2500; 0,3800] | 0,2908 | 164,52 |
| LLM + 8 ejemplos aleatorios | 0,2750 | [0,2150; 0,3400] | 0,2782 | 170,37 |
| **LLM + 8 ejemplos recuperados** | **0,4650** | **[0,3950; 0,5350]** | 0,3839 | 277,56 |
| **Voto mayoritario kNN, sin LLM** | **0,4700** | **[0,4000; 0,5400]** | **0,4152** | **18,58** |
| LLM + demostraciones DSPy | 0,4200 | [0,3549; 0,4900] | 0,3944 | 395,36 |

| Comparación | Δ accuracy | IC 95 % | Significativo | MDE al 80 % |
|---|---:|:--|:--:|---:|
| Ejemplos aleatorios vs. zero-shot | −0,0400 | [−0,0900; +0,0100] | **no** | 0,0715 |
| **Recuperados vs. aleatorios** | **+0,1900** | **[+0,1150; +0,2650]** | **sí** | 0,1072 |
| **Recuperados vs. voto mayoritario kNN** | **−0,0050** | **[−0,0800; +0,0700]** | **no** | 0,1072 |
| Recuperados vs. TF-IDF | −0,0450 | [−0,1150; +0,0250] | **no** | 0,1001 |
| DSPy vs. recuperados | −0,0450 | [−0,1250; +0,0350] | **no** | 0,1144 |

De esa tabla salen cuatro cosas.

**Los ejemplos por sí solos no hacen nada; los ejemplos relevantes lo hacen
todo.** Las demostraciones aleatorias movieron el accuracy en −0,040
(p = 0,134). Las recuperadas ganaron 19 puntos con la misma cantidad. Sin el
brazo aleatorio, «el few-shot ayudó» habría sido indistinguible de «el modelo por
fin vio el formato de salida».

**Y el modelo de lenguaje no es lo que ganó esos 19 puntos.** Un voto mayoritario
sobre los mismos ocho vecinos recuperados —mismo recuperador, mismos ejemplos,
ninguna llamada al LLM— marca 0,4700 contra 0,4650 del LLM y le gana en macro-F1,
a 18,58 ms contra 277,56. Este control no existía antes y debió existir: sin él,
«el few-shot con recuperación funciona» y «el recuperador funciona» son el mismo
número.

No son, eso sí, el mismo predictor. Coinciden en apenas el **44 %** de las
licitaciones, y cada uno acierta donde el otro falla unas 30 veces. El modelo no
está copiando la etiqueta mayoritaria de su prompt: hace algo genuinamente
distinto que aquí no vale nada y cuesta quince veces más.

**La brecha con TF-IDF no se cierra hasta la paridad, y este README decía que
sí.** La recuperación queda 4,5 puntos por debajo del modelo lineal con un
intervalo que cubre el cero — pero con 200 licitaciones esta comparación no podía
haber detectado nada menor a **10 puntos**, y el intervalo sigue siendo
compatible con que el LLM sea **11,5 puntos peor**. Un nulo no es equivalencia.
Cada nulo de la tabla ahora lleva el efecto que sí habría podido encontrar.

**DSPy es más lento que la recuperación, no seis veces más rápido.** Este README
reportaba 412 ms contra 2.362 ms y lo explicaba por las demostraciones fijadas en
tiempo de compilación. Los dos números eran instrumentos y no modelos: el brazo
de recuperación pagaba el timeout de IPv6, y `dspy.LM` cachea por defecto
mientras que ningún otro brazo lo hace. Medidos igual, DSPy cuesta **395 ms
contra 278 ms**. Su primera corrida además marcó 0,1050 por un bug de este
repositorio y no de DSPy, documentado en
[`docs/DECISIONS.md`](docs/DECISIONS.md), sección 5.3.

---

## El hallazgo que le dio forma al proyecto

Cada ítem de línea trae una `description` que parece texto libre:

> `47131803` → "Equipos y suministros de limpieza / Suministros de limpieza /
> Soluciones de limpieza y desinfección / Desinfectantes domésticos"

No es texto libre. Es la ruta de la taxonomía UNSPSC escrita en español. Sobre
una muestra de 444 ítems, el primer campo de esa descripción es una única cadena
constante en **40 de 40** segmentos, y su solapamiento de tokens con la etiqueta
es 1,000.

Entrenar con eso habría producido un puntaje casi perfecto sin aprender nada. El
título escrito por el comprador se solapa con la etiqueta en 0,093, y el 23 % de
los títulos no comparte ni una sola palabra de contenido con ella. Esa es la
tarea real.

`assert_no_leakage()` lanza una excepción si algún campo derivado de la etiqueta
llega a un modelo, y siete tests lo cubren.

Una segunda decisión pesó casi lo mismo. Contar ítems de línea permitía que una
sola licitación de suministro de alimentos, con cientos de líneas, aportara el
61,7 % de una muestra; etiquetar una fila por licitación bajó la clase
mayoritaria a 10,1 %.

El razonamiento completo, incluido un bug de reproducibilidad que el seed no
atrapó, está en [`docs/DECISIONS.md`](docs/DECISIONS.md).

### Y el favorito también perdió

Todas las versiones de este README señalaron a un encoder en español con
fine-tuning como el ganador probable y lo dejaron sin probar. BETO (110 M de
parámetros), con fine-tuning sobre las mismas 2.874 licitaciones, con el número
de épocas elegido en un mes de **validación**:

| Modelo | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| TF-IDF + SVM lineal | 0,5578 | [0,5263; 0,5893] | 0,5077 | 0,26 |
| BETO con fine-tuning | 0,5221 | [0,4916; 0,5546] | 0,4646 | 0,44 |

−0,0357 de accuracy, [−0,0651; −0,0063], p = 0,0196. **La predicción estaba
equivocada**: el transformer no gana y tampoco empata. Pierde, de forma
significativa, después de 144 segundos de entrenamiento en GPU.

> **Corrección.** Una versión anterior de este README reportaba 0,5378 y
> "empata, p = 0,186", eligiendo 15 épocas de una tabla de 6/15/30 cuyos
> accuracy estaban todos medidos sobre **abril, el mes retenido**. Este
> proyecto no tenía ningún conjunto de validación. Eso es selección sobre el
> conjunto de prueba, y de los tres candidatos exactamente uno daba un
> resultado no significativo: justamente el que se publicó.
>
> [`scripts/select_epochs.py`](scripts/select_epochs.py) parte en dos la
> ventana de entrenamiento —entrena en enero–febrero, elige en marzo, lee abril
> una sola vez— y elige **30** épocas, donde BETO pierde significativamente.
> Los dos meses ni siquiera coinciden en la forma de la curva: marzo sube de
> manera monótona a lo largo de toda la rejilla y abril tiene su máximo en 15.
> La "U invertida" era una propiedad del mes desde el que se la leyó.
>
> Lo que sí sobrevive a cualquier elección: **BETO no le gana a TF-IDF en
> ningún punto de la rejilla.** Su mejor marca es −0,0200 y la peor, −0,1134.
> La selección decidió si la derrota se reportaba como significativa, no si
> había derrota. El protocolo completo, la verificación de ruido entre semillas
> y la tabla de test por época —declarada, no usada para elegir— están en
> [`docs/DECISIONS.md`](docs/DECISIONS.md), sección 6.1.

Tres razones por las que no gana, en el orden en que apostaría por ellas: unos 90
ejemplos por clase, títulos de compras públicas que WordPiece destroza (`ADQ.
SERV. REPARACION VEH-.`) donde los n-gramas de caracteres no lo hacen, y una
brecha de macro-F1 más ancha que la de accuracy, lo que apunta a las clases
raras.

### Lo que realmente se pondría en producción

Obligado a responder todo, el encoder acierta el 52,2 %. Si se le permite
abstenerse:

| Confianza ≥ | Cobertura | Accuracy sobre lo que conserva |
|---:|---:|---:|
| 0,50 | 96,2 % | 0,5349 |
| 0,70 | 90,4 % | 0,5552 |
| 0,90 | 80,0 % | 0,5958 |
| 0,95 | 74,5 % | 0,6178 |

La confianza sí ordena las predicciones correctas por encima de las
equivocadas: **AUROC 0,7401, p de permutación 0,0005** sobre 2.000 barajados.
Una versión anterior lo argumentaba desde que la columna sube de forma
monótona, lo cual es casi automático: los conjuntos conservados están anidados,
así que cualquier asociación positiva, por leve que sea, produce una columna
creciente. El AUROC es la prueba; la monotonía no lo era.

Aun así no alcanza: 61,8 % de accuracy absteniéndose en un cuarto de las
licitaciones significa que casi cuatro de cada diez de las conservadas están
mal. **Esto es una herramienta de sugerencia, no un sistema de clasificación
automática**, y decir lo contrario sería la exageración más fácil de todo el
proyecto.

Vale la pena nombrar un costo de la corrección anterior. Con las 15 épocas
elegidas sobre test, esta tabla era *mejor*: 39,7 % de cobertura con 0,7275 de
accuracy. Treinta épocas compran accuracy en el papel y gastan calibración, y
el número de épocas se eligió solo por accuracy. Volver a elegir ahora con un
criterio sensible a la calibración, después de ver cuál favorece a esta tabla,
sería el error original con un tercer disfraz.

### Cuantizado para CPU: la línea de accuracy esconde casi todo el cambio

El encoder es el modelo que una mesa de compras correría, y lo correría sin GPU.
Tres brazos, los mismos ítems de prueba en el mismo orden, un hilo de CPU, batch
de tamaño uno:

| Brazo | Accuracy | Macro-F1 | Tamaño | p50 | **Tasa de flips** | KL media |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0,5231 | 0,4657 | 439,5 MB | 108,5 ms | — | — |
| ONNX FP32 | 0,5231 | 0,4657 | 439,7 MB | 82,2 ms | **0,00 %** | −0,00000 |
| ONNX INT8 | 0,5179 | **0,4741** | **110,6 MB** | **27,7 ms** | **6,83 %** | 0,09224 |

Exportar a ONNX no cambia absolutamente nada: cero flips, KL cero, 1,32× más
rápido. INT8 es 4,0× más chico, 3,9× más rápido, y su caída de accuracy de
0,0053 tiene un intervalo bootstrap de [−0,0168; +0,0063] que cubre el cero.
Reportado de la manera habitual, eso se lee como gratis.

No es gratis. **65 de 952 predicciones cambiaron, 6,83 %, mientras el accuracy se
movió medio punto.**

| Dirección | Cantidad |
|---|---:|
| Mal → bien | 15 |
| Bien → mal | 20 |
| Mal → *otra* respuesta mal | **30** |

Casi la mitad de los flips están en esa última fila y ninguna métrica agregada
puede verlos: el accuracy compensa las dos primeras entre sí y es ciega a la
tercera. El encuadre sigue a [*Accuracy Is Not All You
Need*](https://arxiv.org/abs/2407.09141) de Microsoft Research, que argumenta que
los flips y la divergencia KL son la forma correcta de comparar un modelo
comprimido con su base.

> **Corrección.** Una versión anterior le ponía número a ese contraste: «15,6
> veces más churn del que sugiere el número publicado», dividiendo la tasa de
> flips por el movimiento neto de accuracy. Ese cociente no es una cantidad. Su
> denominador es una diferencia entre dos números casi iguales cuyo intervalo
> cubre el cero, así que el cociente no está acotado — un denominador en
> cualquiera de los extremos de ese intervalo lo manda al infinito o le cambia
> el signo. El contraste es real; la aritmética que lo vestía, no.

INT8 es igual lo que hay que desplegar acá. Simplemente no es el mismo modelo:
coincide con el original el 93 % de las veces y es igual de bueno en promedio, y
de esa frase normalmente solo se escribe la segunda mitad. Además tiene *mejor*
macro-F1 que el modelo que comprime, lo que recuerda cuánto trabajo está haciendo
el «igual de bueno en promedio».

---

## Datos

API OCDS de ChileCompra / Mercado Público. **CC0**, sin key, sin registro.

| | |
|---|---:|
| Licitaciones con etiqueta UNSPSC | 3.826 |
| Ítems de línea | 15.823 |
| Segmentos distintos | 54 |
| Licitaciones de una sola categoría | 91,6 % |
| Largo medio del título | 8,3 palabras |
| Meses cubiertos | Ene–Abr 2026 |

Una rareza que vale la pena conocer: la API devuelve sus propias URLs de detalle
sobre `http://`, y esas conexiones se cierran sin respuesta. Cada URL se
reescribe a `https://` antes de usarla. Un crawler que confía en el payload falla
en todas las peticiones.

---

## Cómo ejecutarlo

```bash
pip install -e ".[dev]"
python -m pytest                                # 83 tests
python scripts/download_data.py                 # ~11 minutos
python scripts/run_experiment.py --skip-llm     # segundos
python scripts/run_experiment.py                # necesita Ollama, ~3 minutos
python scripts/measure_llm_overhead.py          # latencia del cliente vs. del modelo
python scripts/run_llm_variants.py --sample 200 # brazos few-shot, ~8 minutos
python scripts/select_epochs.py --sensitivity   # épocas por validación, GPU
python scripts/run_encoder.py                   # BETO, GPU, ~3 minutos
python scripts/run_quantisation.py              # ONNX + INT8, tiempos de CPU
```

El crawl queda cacheado como JSON y cargado en un único archivo DuckDB, así que
todo lo posterior a la primera descarga corre sin conexión. Las preguntas sobre
el corpus en `download_data.py` (distribución de etiquetas, concentración de
compradores, deriva mes a mes) son SQL, porque eso es lo que son.

Dos de estos scripts existen por defectos encontrados después de publicar.
`measure_llm_overhead.py` separa la latencia del modelo de la del cliente HTTP, y
`select_epochs.py` elige el número de épocas del encoder en un mes de validación
en vez de en el mes retenido.

Cada cifra de este README sale de un archivo en `reports/`, todos producidos por
los scripts de arriba: `metrics_corpus.json`, `metrics_experiment.json`,
`metrics_llm_variants.json`, `metrics_encoder.json`, `metrics_quantisation.json`,
`metrics_epoch_selection.json` y `metrics_llm_overhead.json`.
`tests/test_published_numbers.py` falla si una cifra de la prosa no está en
alguno de ellos.

---

## Limitaciones

**Sin GEPA.** DSPy 3.3 incluye `dspy.GEPA`, del que se reporta que alcanza una
calidad dada con 35× menos rollouts. Solo se corrió `BootstrapFewShot`, así que
no se afirma nada sobre optimizadores más fuertes.

**Sin calibración.** Los puntajes softmax del encoder ordenan bien pero no están
calibrados, así que la tabla de cobertura entrega puntos de operación y no tasas
de error garantizadas. Temperature scaling o predicción conforme es lo más útil
que quedó sin hacer.

**Solo segmento.** UNSPSC anida segmento → familia → clase → producto. Solo se
predice el segmento de dos dígitos; la familia de cuatro dígitos es más difícil y
más útil.

**La clase `other` es la mayor fuente individual de error.** Son 22 segmentos
raros metidos en una bolsa, así que no tiene un vocabulario coherente. Reportar
el accuracy sin decirlo exageraría cuánto del error es confusión genuina.

**Cuatro meses.** Sin estacionalidad, y el drift a lo largo de un año queda sin
probar.

**Los brazos few-shot corren sobre 200 de las 952 licitaciones de prueba**, por
tiempo. Ese subconjunto es lo bastante chico como para que la comparación contra
TF-IDF no pueda detectar una brecha menor a unos 10 puntos de accuracy, y por eso
esos nulos se reportan con su efecto mínimo detectable y no como empates. El
brazo zero-shot ahora corre sobre las 952.

**El número de épocas se elige sobre un corpus más chico que aquel al que se
aplica.** La validación entrena con 1.922 licitaciones y el reajuste final con
2.874, así que 30 épocas es lo mejor para el más chico. Igualar *pasos* de
optimización en vez de épocas eliminaría eso; no está hecho. El sesgo va en
contra del modelo, no a favor.

**El brazo LLM no es reproducible bit a bit.** La temperatura es cero, pero el
accuracy zero-shot de Ollama se movió de 0,2767 a 0,2794 entre dos corridas de
código idéntico sobre datos idénticos. Diferencias menores a medio punto en
cualquier fila de LLM no deberían leerse como reales.

## Licencia

MIT para el código; los datos son CC0 de ChileCompra.
