# Predicción de categorías de compra a partir de títulos de licitaciones públicas chilenas

[![CI](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
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

**En zero-shot, no lo hace**: el LLM pierde por 32 puntos de accuracy y tarda
9.345 veces más. Con ocho ejemplos *recuperados* empata, y sigue siendo unas
6.400 veces más lento. Ocho ejemplos *aleatorios* no cambian nada, que es
justamente el resultado que el brazo de control existe para encontrar.

---

## Resultados

Conjunto de prueba: abril de 2026 (952 licitaciones). Entrenado con enero a marzo
(2.874). 32 categorías tras agrupar la cola de clases raras.

| Modelo | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| Clase mayoritaria | 0,1429 | [0,1208; 0,1670] | 0,0078 | 0,00 |
| Reglas por palabras clave | 0,2384 | [0,2111; 0,2658] | 0,1295 | 0,08 |
| **TF-IDF + SVM lineal** | **0,5578** | **[0,5263; 0,5893]** | **0,5077** | **0,24** |
| LLM local (qwen2.5:7b) | 0,2767 | [0,2267; 0,3300] | 0,2248 | 2.242,89 |

Bootstrap pareado sobre 5.000 remuestreos, en las filas que ambos modelos vieron:

| Comparación | Δ accuracy | IC 95 % | Significativo |
|---|---:|:--|:--:|
| TF-IDF vs. palabras clave | +0,3193 | [+0,2805; +0,3571] | sí |
| **LLM vs. TF-IDF** | **−0,3200** | **[−0,3900; −0,2500]** | **sí** |

El modelo supervisado aprende el vocabulario de las compras públicas chilenas a
partir de 2.874 ejemplos. En zero-shot, el LLM tiene que razonarlo desde un
prompt, y ocho palabras abreviadas en mayúsculas no dan mucho material para
razonar.

### Después se le dieron ejemplos

Mismo mes de prueba, mismo subconjunto de 200 licitaciones con seed fijo, k = 8:

| Variante | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| TF-IDF + SVM lineal | 0,5100 | [0,4400; 0,5800] | 0,4400 | 0,36 |
| LLM zero-shot | 0,3000 | [0,2350; 0,3650] | 0,2823 | 2.249 |
| LLM + 8 ejemplos aleatorios | 0,2550 | [0,1950; 0,3200] | 0,2532 | 2.245 |
| **LLM + 8 ejemplos recuperados** | **0,4700** | **[0,4000; 0,5400]** | 0,3852 | 2.362 |
| LLM + demostraciones DSPy | 0,4200 | [0,3549; 0,4900] | **0,3944** | **412** |

| Comparación | Δ accuracy | IC 95 % | Significativo |
|---|---:|:--|:--:|
| Ejemplos aleatorios vs. zero-shot | −0,0450 | [−0,0950; +0,0050] | **no** |
| **Recuperados vs. aleatorios** | **+0,2150** | **[+0,1400; +0,2900]** | **sí** |
| Recuperados vs. TF-IDF | −0,0400 | [−0,1150; +0,0350] | **no** |
| DSPy vs. recuperados | −0,0500 | [−0,1300; +0,0300] | **no** |

De esa tabla salen tres cosas.

**Los ejemplos por sí solos no hacen nada; los ejemplos relevantes lo hacen
todo.** Las demostraciones aleatorias movieron el accuracy en −0,045
(p = 0,086). Las recuperadas ganaron 21,5 puntos con la misma cantidad. Sin el
brazo aleatorio, «el few-shot ayudó» habría sido indistinguible de «el modelo por
fin vio el formato de salida».

**La brecha se cierra hasta la paridad estadística.** El few-shot con
recuperación queda 4 puntos por debajo del modelo lineal, con un intervalo que
cubre el cero. Empata, a unas 6.400 veces el costo por licitación.

**DSPy iguala a la recuperación con un sexto de la latencia** (412 ms frente a
2.362 ms), porque sus demostraciones quedan fijas en tiempo de compilación en vez
de reconstruirse en cada consulta. Su primera corrida marcó 0,1050, y la causa
fue un bug de este repositorio y no de DSPy: a la signature se le entregaron
códigos numéricos pelados mientras que todos los demás brazos recibían los
nombres de categoría en español. Está documentado en
[`docs/DECISIONS.md`](docs/DECISIONS.md), sección 5.3, porque «la herramienta
rindió mal» y «le di a la herramienta un problema peor» se ven idénticos desde
afuera.

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
parámetros), con fine-tuning sobre las mismas 2.874 licitaciones:

| Modelo | Accuracy | IC 95 % | Macro-F1 | ms por licitación |
|---|---:|:--|---:|---:|
| TF-IDF + SVM lineal | 0,5578 | [0,5263; 0,5893] | 0,5077 | 0,31 |
| BETO con fine-tuning | 0,5378 | [0,5063; 0,5704] | 0,4886 | 0,50 |

−0,0200 de accuracy, [−0,0494; +0,0095], p = 0,186. **La predicción estaba
equivocada**: el transformer empata, no gana, después de 80 segundos de
entrenamiento en GPU contra 2,5 segundos de ajuste en CPU.

La primera respuesta estaba equivocada en la dirección contraria. Con las 6
épocas configuradas, BETO marcó 0,4989 y perdió de forma significativa, con la
pérdida de entrenamiento todavía en 1,865. Con 30 épocas hace overfitting y vuelve
a bajar a 0,5221. Solo el centro de esa U invertida es la respuesta real, y
cualquiera de los dos extremos habría dado un titular seguro y equivocado.

Tres razones por las que no gana, en el orden en que apostaría por ellas: unos 90
ejemplos por clase, títulos de compras públicas que WordPiece destroza (`ADQ.
SERV. REPARACION VEH-.`) donde los n-gramas de caracteres no lo hacen, y una
brecha de macro-F1 más ancha que la de accuracy en todos los conteos de épocas,
lo que apunta a las clases raras.

### Lo que realmente se pondría en producción

Obligado a responder todo, el encoder acierta el 53,8 %. Si se le permite
abstenerse:

| Confianza ≥ | Cobertura | Accuracy sobre lo que conserva |
|---:|---:|---:|
| 0,50 | 85,8 % | 0,5900 |
| 0,70 | 72,2 % | 0,6390 |
| 0,90 | 50,6 % | 0,7054 |
| 0,95 | 39,7 % | 0,7275 |

El accuracy sube de forma monótona con el umbral, así que la confianza lleva
señal real. Aun así no alcanza: 72,8 % de accuracy sobre el 39,7 % de las
licitaciones significa que casi tres de cada diez de las conservadas están mal.
**Esto es una herramienta de sugerencia, no un sistema de clasificación
automática**, y decir lo contrario sería la exageración más fácil de todo el
proyecto.

### Cuantizado para CPU: la línea de accuracy esconde casi todo el cambio

El encoder es el modelo que correría una mesa de compras, y correría sin GPU.
Tres brazos, los mismos ítems de prueba en el mismo orden, un solo hilo de CPU,
batch de tamaño uno:

| Brazo | Accuracy | Tamaño | p50 | **Tasa de flips** | KL medio |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0,5399 | 440 MB | 116,7 ms | — | — |
| ONNX FP32 | 0,5399 | 440 MB | 90,7 ms | **0,00 %** | 0,00000 |
| ONNX INT8 | 0,5347 | **111 MB** | **29,2 ms** | **8,19 %** | 0,06567 |

Exportar a ONNX no cambia absolutamente nada: cero flips, KL exactamente cero,
1,29× más rápido. INT8 es 4× más pequeño, 4× más rápido, y su caída de accuracy
de 0,0053 tiene un intervalo bootstrap de [−0,0179; +0,0074] que cubre el cero.
Reportado de la forma habitual, eso se lee como gratis.

No es gratis. **78 de 952 predicciones cambiaron, un 8,19 %, frente a un
movimiento neto de accuracy de −0,53 % — 15,6 veces más rotación de la que
implica el número publicado.**

| Dirección | Conteo |
|---|---:|
| Incorrecto → correcto | 17 |
| Correcto → incorrecto | 22 |
| Incorrecto → otra respuesta incorrecta *distinta* | **39** |

La mitad de los flips está en esa última fila y ninguna métrica agregada los ve:
el accuracy compensa las dos primeras entre sí y es ciego a la tercera. El
encuadre sigue a [*Accuracy Is Not All You
Need*](https://arxiv.org/abs/2407.09141), de Microsoft Research, que argumenta
que los flips y la divergencia KL son la forma correcta de comparar un modelo
comprimido con su baseline.

INT8 sigue siendo lo que hay que poner en producción acá. Simplemente no es el
mismo modelo: coincide con el original el 92 % de las veces y es igual de bueno
en promedio, y normalmente solo se escribe la segunda mitad de esa frase.

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
python -m pytest                                     # 35 tests
python scripts/download_data.py                      # ~11 minutos
python scripts/run_experiment.py --skip-llm          # segundos
python scripts/run_experiment.py --llm-sample 300    # necesita Ollama, ~11 minutos
python scripts/run_llm_variants.py --sample 200      # brazos few-shot, ~35 minutos
```

El crawl se cachea como JSON y se carga en un único archivo DuckDB, así que todo
lo posterior a la primera descarga corre offline. Las preguntas sobre el corpus
en `download_data.py` (distribución de etiquetas, concentración de compradores,
drift por mes) están en SQL, porque eso es lo que son.

Todos los números de este README salen de `reports/metrics_corpus.json`,
`reports/metrics_experiment.json` y `reports/metrics_llm_variants.json`,
todos producidos por los scripts de arriba.

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

**El LLM corrió sobre 300 de 952 licitaciones de prueba**, por tiempo. Su
intervalo es más ancho en consecuencia.

## Licencia

MIT para el código; los datos son CC0 de ChileCompra.
