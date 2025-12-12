# main.py
import numpy as np
import time
from SHADE import SHADE
from funcionFitness import evaluar_disenio

# Parámetros de optimización (5 parámetros)
# PARÁMETROS FÍSICOS A OPTIMIZAR
# [a, b, d, h, c]
# ----------------------------------------------------------
# a = profundidad guía
# b = ancho guía
# d = separación entre guías principales
# h = ancho del brazo
# c = separación entre brazos


dimension = 5
lower = [2.0, 1.0, 0.5, 0.2,0.5]   # [a,b, d, h, c]
upper = [3.0, 1.5, 1.2, 0.5,1.2 ]

# Control de corridas completas
num_runs = 1          # cuántas veces repetir toda la optimización 
pop_size = 10
maxIter = 4

resultados = []


shade = SHADE(
    dimension=dimension,
    pop_size=pop_size,
    lower=lower,
    upper=upper,
    function=evaluar_disenio,   # esta función ejecuta la simulación y devuelve fitness
    maxIter=maxIter,
    H=10,
    p=0.2
)

t0 = time.time()
best_sol, best_val, hist_best, hist_mean, hist_div, iter_to_min = shade.run(verbose=True)
t1 = time.time()

print(f"\n terminado en {t1-t0:.1f} s")
print("Mejor solución:", best_sol)
print("Mejor fitness:", best_val)
resultados.append({
   
    "best_sol": best_sol,
    "best_val": best_val,
    "time_s": t1-t0,
    "iter_to_min": iter_to_min,
    "history_best": hist_best
})

# Guardar resultados
np.save("resultados_optimizacion.npy", resultados)
print("\nTodos los runs completos. Resultados guardados en resultados_optimizacion.npy")



