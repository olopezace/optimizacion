# Guía de uso

> Para pegar en `README.md`, reemplazando el bloque «Comandos de inicio».
> El resto del README documenta la función *fitness* y se conserva tal cual.

## Qué hay en el repositorio

| Carpeta | Para qué sirve |
|---|---|
| `optimizacion completa/` | Optimizador secuencial. Evalúa la población de uno en uno. |
| `Optimización_Paralela/` | Optimizador paralelo. Evalúa toda la población a la vez, una simulación por individuo. |
| `Resultados/` | Guiones de posproceso y figuras de las corridas ya hechas. |
| `DE/` | Evolución diferencial clásica y su validación sobre funciones analíticas. |
| `Función_fitness/` | Cuaderno que ejercita la *fitness* sobre espectros sintéticos, sin MEEP. |
| `Intrucciones/` | Instalación en Windows mediante WSL. |

Los cuatro archivos de cada carpeta de optimización son intercambiables entre sí:
`main.py` fija los parámetros y lanza; `SHADE.py` es el algoritmo; `funcionFitness.py`
traduce la respuesta en un escalar; `geometria_sinpng_*.py` es el modelo de MEEP.

## Instalación

```bash
# (solo en Windows) subsistema Linux
wsl --install

# entorno con MEEP paralelo
conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*
conda activate pmp

# comprobación: no debe dar error
python -c "import meep; print(meep.__version__)"
```

## Las dos maneras de usar los núcleos

Es la distinción más importante de esta guía, porque el mismo `mpirun` hace dos cosas
distintas según la carpeta en la que se ejecute.

**Descomposición de dominio** (`optimizacion completa/`). Los *N* procesos se reparten
la celda computacional y cooperan en **una sola** simulación. Acorta cada evaluación,
pero la población se sigue recorriendo de una en una.

**Paralelismo por tareas** (`Optimización_Paralela/`). Cada proceso ejecuta **una
simulación completa e independiente**, una por individuo de la población. No acorta
cada evaluación —al contrario, cada una corre en un solo núcleo y tarda más— pero
resuelve la generación entera de una vez.

Para una corrida de optimización real conviene el segundo. El primero es útil para
depurar, para evaluar un diseño suelto o cuando la población es más pequeña que el
número de núcleos.

## Optimizador secuencial

```bash
cd "optimizacion completa"
python main.py                 # todo en un núcleo
mpirun -np 16 python main.py   # una simulación repartida entre 16 núcleos
```

Parámetros, todos en `main.py`: `lower` y `upper` (límites del espacio de búsqueda),
`pop_size`, `maxIter`, `num_runs`, y `H` y `p` en la llamada a `SHADE`.
Salida: `resultados_optimizacion.csv` con la mejor solución de cada corrida.

> **Antes de usarlo, lea «Problemas conocidos» al final.** Esta carpeta no se ejecuta
> tal cual en la versión actual.

## Optimizador paralelo

```bash
cd "Optimización_Paralela"
mpirun -np 24 python -m mpi4py main.py
```

Tres detalles que importan:

- **`-m mpi4py` no es opcional.** Instala un manejador que aborta el trabajo MPI
  completo si un proceso muere. Sin él, la muerte de un proceso deja a los demás
  esperando y la corrida se cuelga en lugar de fallar.
- **`-np` debe coincidir con `pop_size`.** El optimizador no avanza hasta que toda la
  generación está evaluada, de modo que una población que no sea múltiplo del número
  de procesos deja parte de la máquina ociosa en la última oleada. Con `pop_size = 24`,
  use `-np 24`.
- **No pase del número de núcleos *físicos*.** Los núcleos lógicos de *Hyper-Threading*
  no añaden ancho de banda de memoria, que es lo que limita a FDTD. Compruébelo con
  `lscpu`.

Salidas:

- `particulas_optimizacion.csv` — una fila por evaluación, con los cinco parámetros y
  el fitness. Se escribe incrementalmente, de modo que la corrida puede inspeccionarse
  mientras avanza y sobrevive a una interrupción. Es además el conjunto de datos para
  trabajo posterior.
- `resultados_optimizacion.csv` — resumen de cada corrida completa.

## Parámetros que se tocan con más frecuencia

| Qué se cambia | Dónde | Nota |
|---|---|---|
| Límites del espacio de búsqueda | `lower`, `upper` en `main.py` | Los vectores se recortan a esta caja. |
| Tamaño de población e iteraciones | `pop_size`, `maxIter` en `main.py` | El número de simulaciones es `pop_size * (1 + maxIter)`. |
| Comportamiento de SHADE | `H`, `p` en la llamada a `SHADE` | El conjunto de élite tiene un piso de dos individuos, así que `p` por debajo de `2/pop_size` no cambia nada. |
| Banda de frecuencias | `fminima`, `fmaxima` en el guion de geometría | Actualice también `bands` en `funcionFitness.py`. |
| Resolución espacial | `resolution` en el guion de geometría | Es lo que más pesa en el tiempo de cálculo. |
| Pesos y λ | `W`, `lambda_reg` en `funcionFitness.py` | Los pesos se normalizan: importan las proporciones. |

## Cuánto tarda

Medido sobre un servidor de 24 núcleos físicos y 64 GB, con una malla de unos
17 millones de puntos a resolución 40:

| Esquema | Tiempo por evaluación |
|---|---|
| Un solo núcleo | ≈ 36 min |
| Descomposición de dominio, 16 procesos | ≈ 12 min |
| Paralelismo por tareas, 24 procesos | ≈ 199 min de reloj por generación de 24, es decir ≈ 8.3 min amortizados |

El rendimiento del esquema por tareas se satura en torno a 16 simulaciones
concurrentes: por encima de eso los procesos se estorban compitiendo por el ancho de
banda de memoria. Para escalar más hacen falta más nodos, no más núcleos.

## Problemas conocidos

**`optimizacion completa/` no se ejecuta en la versión actual.** Son cuatro arreglos
pequeños, todos ya resueltos en `Optimización_Paralela/`, de donde pueden copiarse:

1. `geometria_sinpng_m.py:344` devuelve `deltaA`, cuya asignación está comentada en la
   línea 291 → `NameError`.
2. Ese mismo `return` entrega cinco valores, pero `funcionFitness.py:169` desempaqueta
   tres → `ValueError`.
3. `funcionFitness.py:100` usa `np.trapz`, retirado en NumPy 2.0 → `AttributeError`.
   Sustitúyalo por `np.trapezoid`.
4. `main.py` guarda con `np.savetxt` una lista de diccionarios, que esa función no sabe
   escribir → `TypeError`. Use `np.save` y la extensión `.npy`, como hace la versión
   paralela.

**La partición en regiones no coincide con la banda simulada.** `funcionFitness.py`
define `bands = [(65,70), (71,110), (111,115)]`, heredado del caso completo de la banda
2+3 de ALMA, mientras que la simulación va de 75 a 110 GHz. Las regiones baja y alta
quedan sin puntos de frecuencia: su contribución es nula pero su penalización es
máxima. Conviene redefinirlas dentro de la banda que se simule.

**La métrica de oscilación mide otra cosa.** Se calcula como el valor absoluto de la
integral de la segunda derivada, que por el teorema fundamental del cálculo se reduce a
la diferencia de pendientes entre los extremos del intervalo. Dos curvas con las mismas
pendientes en los extremos reciben el mismo valor aunque una oscile y la otra sea plana.
Integrar el valor absoluto de la segunda derivada sí las distingue. Como `λ = 0.008`
reduce mucho el peso de esta componente, el efecto sobre los diseños es pequeño.

**El criterio de convergencia no aplica a esta *fitness*.** La variable que registra la
iteración del mejor individuo compara contra una tolerancia positiva, criterio
apropiado para las funciones de prueba de mínimo nulo con que se validó el algoritmo.
Como esta *fitness* es negativa por construcción, se satisface desde la primera
iteración. Para saber cuándo se estabilizó el mejor individuo, tome el mínimo del
historial.

# Función *fitness* para el diseño del híbrido en cuadratura (ALMA Band 2+3)

Este documento describe **con precisión y con ejemplos** la función *fitness* usada para optimizar un híbrido en cuadratura mediante PSO (Particle Swarm Optimization). Se aclaran las matrices y vectores utilizados en la Sección 7.5 del documento técnico, se explica el **significado de las filas y columnas** de cada matriz y se detalla **qué es exactamente la matriz (vector) _F_**. Además, se indica **dónde y cómo ajustar los pesos** en el repositorio asociado y el **valor por defecto de la ponderación de oscilaciones** (λ).

> Repositorio de referencia: `PSO_ALMA_BAND2-3_QuadratureHybrid`  
> Documento: *Quadrature Hybrid Optimization for ALMA* 


## 1) Variables, regiones y restricciones evaluadas

### 1.1 Regiones de frecuencia
Para capturar comportamientos distintos a lo largo del ancho de banda, los datos se **dividen en 3 sub-bandas** (ajustables):
- **Región 1 (R1)**: 65–70 GHz (baja)  
- **Región 2 (R2)**: 71–110 GHz (media)  
- **Región 3 (R3)**: 111–115 GHz (alta)

> Estas fronteras se fijaron por experiencia; pueden modificarse si se desea explorar otra partición del ancho de banda.

### 1.2 Métricas/Restricciones evaluadas (cinco filas)
En **cada región** se evalúan **cinco** magnitudes clave (una por **fila** de las matrices que se describen más abajo):

1. **S31** (acoplamiento al puerto 3) – error suave (cercanía a −3 dB y cumplimiento del rango [−3.5, −2.5] dB).  
2. **S21** (acoplamiento al puerto 2) – error suave (mismo criterio que S31).  
3. **ΔA** (desbalance de amplitud) – error suave (objetivo típico: ΔA < 1 dB en banda).  
4. **Overshoot de S31** – **componente oscilatoria** (picos/ondulaciones indeseadas).  
5. **Overshoot de S21** – **componente oscilatoria**.

> Notar que **S11, S41 y Δφ** no entran directamente en la *fitness*; se inspeccionan luego, pues suelen mejorar al optimizar S31/S21 y ΔA.


## 2) Matrices de **Pesos** y de **Penalizaciones**

### 2.1 Matriz de pesos base **W** (5×3)
\
Filas = las 5 métricas anteriores.  
Columnas = las 3 regiones (R1, R2, R3).

```
W = [ [w11, w12, w13],   ← fila 1: S31
      [w21, w22, w23],   ← fila 2: S21
      [w31, w32, w33],   ← fila 3: ΔA
      [w41, w42, w43],   ← fila 4: Overshoot S31
      [w51, w52, w53] ]  ← fila 5: Overshoot S21
```

**Interpretación:** `wij` es la **importancia relativa** que se le otorga a la **métrica i** en la **región j**. Por ejemplo, puede interesar **enfatizar** la banda media suave asignando `w12` y `w22` más altos, y **penalizar** las oscilaciones en los extremos dando más peso a `w41,w43,w51,w53`.

> En el repositorio, estos pesos se configuran en el archivo de la *fitness* https://github.com/Jorgecardenas1/PSO_ALMA_BAND2-3_QuadratureHybrid/blob/master/PSO/fitness_func.py.

Usados en la simulación 

```
W = [ [0.005, 0.470, 0.001],   ← fila 1: S31
      [0.005, 0.470, 0.001],   ← fila 2: S21
      [0.001, 0.005, 0.001],   ← fila 3: ΔA
      [0.001, 0.005, 0.001],   ← fila 4: Overshoot S31
      [0.001, 0.032, 0.001] ]  ← fila 5: Overshoot S21
```

### 2.2 Matriz de **Penalizaciones** (5×3)
Tiene **misma forma** (5×3) y se construye **a partir del cumplimiento de las restricciones por punto de frecuencia** dentro de cada región. Para cada métrica/región se genera una **secuencia binaria** (1 si cumple, 0 si viola), y se condensa en un **valor real** en [0,1] que expresa qué tanto se incumple. Ejemplos típicos:
- Para S31 y S21: `penalty = 1 - promedio(secuencia_binaria)` (cero si siempre cumple; cercano a 1 si suele violar).
- Para Overshoot y ΔA: se usan funciones de conteo/umbral similares que arrojan un **grado de penalización** (≥0).

Denotemos esta matriz como:

```
Penalties = [ [p11, p12, p13],
              [p21, p22, p23],
              [p31, p32, p33],
              [p41, p42, p43],
              [p51, p52, p53] ]git pul
```

### 2.3 Pesos penalizados y normalización
Primero se amplifican los pesos base (por ejemplo por 100) y se **suman** las penalizaciones:

```
W_penalized_raw = 100 * W + Penalties
```

Luego se **normaliza a [0,1]** para evitar escalas dispares:

```
wmax = max(W_penalized_raw)
wmin = min(W_penalized_raw)
W_penalized = (W_penalized_raw - wmin) / (wmax - wmin)
```

**Resultado:** `W_penalized` es 5×3 y combina prioridad (W) + castigo por incumplimiento (Penalties), ya **escalado**.


## 3) ¿Qué es exactamente **F**? (y cómo se calcula)

**F no es una “matriz de diseño”**, sino una **matriz de medidas (5×3)**, una por **métrica** (fila) y **región** (columna). Cada entrada `fij` es un **número real** que resume el **error o la intensidad de oscilación** de la métrica *i* en la región *j*:

```
F = [ [f11, f12, f13],   ← S31: error por región (p.ej. MSE vs. valor ideal)
      [f21, f22, f23],   ← S21: error por región
      [f31, f32, f33],   ← ΔA:  error por región
      [f41, f42, f43],   ← Overshoot S31: intensidad de oscilación por región
      [f51, f52, f53] ]  ← Overshoot S21: intensidad de oscilación por región
```

Cálculo de cada bloque:
- **Errores “suaves” (S31, S21, ΔA)** → se condensan como **MSE** frente al perfil objetivo (p. ej., −3 dB o ΔA≈0):  
  `fij = MSE( Señal_simulada(i, región j), Señal_objetivo(i) )`.
- **Oscilaciones (Overshoot)** → se cuantifican mediante la **segunda derivada** respecto a la frecuencia y una **integral de magnitud** para obtener un escalar por región:  
  1) `Y = d²/d f² ( Señal_simulada )`  
  2) `fij = | ∫_región Y(f) df |`

> En la práctica se implementa con derivada numérica (segundas diferencias) y una suma trapezoidal en la sub-banda.


## 4) Combinación de F con los pesos (qué se multiplica con qué)

Para combinar pesos y medidas por **métrica** y **región** y obtener un **costo por métrica**, se calcula el producto
**por filas** y luego se selecciona la **diagonal** del resultado. La manera consistente de hacerlo es:

```
# Dimensiones: W_penalized (5×3) · F^T (3×5) → M (5×5)
M = W_penalized · F^T
```

La **diagonal** de `M` contiene, para cada fila *i*, la **suma ponderada por regiones** `∑_j wij · fij`:

```
diag_i = (M)_ii = ∑_{j=1..3}  W_penalized[i,j] * F[i,j]     (i = 1..5)
```

Para **estabilizar** rangos (valores muy grandes/pequeños) se aplica un **escalado logarítmico** al producto antes de tomar la diagonal. Una implementación típica es:

```
M_log = log( 100 * M )
d = Diag( M_log )   # vector d de tamaño 5
```

> `d = [d1, d2, d3, d4, d5]` representa el **costo por métrica** a lo largo de todo el ancho de banda, ya ponderado y escalado.


## 5) Mezcla “suave vs. oscilatorio” y valor de **λ**

La *fitness* final separa la contribución **suave** (S31, S21, ΔA) de la **oscilatoria** (Overshoot). Sea `d = [d1..d5]`:

```
fitness = (d1 + d2 + d3)  +  λ * (d4 + d5)
```

- Los **primeros 3 términos** capturan cercanía a −3 dB y ΔA bajo control.  
- Los **últimos 2 términos**, escalados por **λ**, penalizan ondulaciones/picos.

**Valor recomendado de λ:** `λ = 0.008` (valor que ofreció el mejor balance en experimentación).  
Este parámetro **se puede ajustar** si se desea enfatizar más/menos la suavidad frente a la oscilación.


## 6) Ejemplo numérico mínimo (ilustrativo)

Suponga, para **S31** (fila 1), estos números por región (ya calculados):  
`F[1,:] = [0.020, 0.010, 0.025]`  (MSE en R1, R2, R3).  
Pesos penalizados: `W_penalized[1,:] = [0.6, 0.9, 0.7]`.

Entonces el costo para **S31** es:
```
d1 = 0.6*0.020 + 0.9*0.010 + 0.7*0.025 = 0.012 + 0.009 + 0.0175 = 0.0385
```
(En la implementación real, esta suma sale de la **diagonal** de `M = W_penalized · F^T`, tras el **log-escalado**).

Se repite para S21 (d2), ΔA (d3), Overshoot S31 (d4) y Overshoot S21 (d5), y finalmente:
```
fitness = (d1 + d2 + d3) + λ*(d4 + d5)
```


## 7) Resumen conceptual (una mirada de 10 segundos)

- **W (5×3)**: prioridades por **métrica** (filas) y **región** (columnas).  
- **Penalties (5×3)**: cuánto se **incumple** cada restricción por región; se suma a `W` (amplificado) y se normaliza → `W_penalized`.  
- **F (5×3)**: **medidas** por métrica/región (MSE para S31/S21/ΔA y “intensidad de oscilación” para overshoot).  
- Producto **por métrica**: `diag( log(100 * (W_penalized · F^T)) )  →  d ∈ ℝ^5`.  
- **Fitness**: `(d1+d2+d3) + λ*(d4+d5)` con `λ ≈ 0.008` por defecto.  
- **Dónde editar**: `PSO/fitness_func.py` (pesos, penalizaciones, λ y límites por región).


## 8) Notas prácticas y recomendaciones

- Si quiere **priorizar** aún más la **parte suave** en banda media, aumente `w12` y `w22` (y quizá `w32`), y/o reduzca `λ`.  
- Si observa **picos** en los extremos de banda, incremente `w41, w43, w51, w53` y/o suba levemente `λ`.  
- La **partición de banda** (R1/R2/R3) puede mover la frontera de evaluación de picos para que la métrica de oscilación “mire” donde realmente aparecen.  
- Si hay **sobre-penalización** (fitness saturada), revise la normalización de `W_penalized` y los **umbrales** con los que se convierte la secuencia binaria a penalizaciones.


---

### Referencias y recursos
- Repositorio: https://github.com/Jorgecardenas1/PSO_ALMA_BAND2-3_QuadratureHybrid  
- Documento (Sección 7.5 – función *fitness*; construcción de `W`, `Penalties`, `F` y mezcla con `λ`): *Quadrature Hybrid Optimization for ALMA* (PDF provisto).

