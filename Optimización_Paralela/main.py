# main.py
import numpy as np
import time
from SHADE import SHADE
from funcionFitness import evaluar_disenio, evaluar_disenio_batch
from mpi_parallel_eval import rank, worker_loop, stop_workers
import meep as mp
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
pop_size = 24
maxIter = 100

resultados = []

if rank == 0:
    # ------------------------------------------------------------
    # Proceso maestro: corre SHADE con total normalidad. Las únicas
    # dos líneas que cambian frente a tu versión original son el
    # import de evaluar_disenio_batch (arriba) y
    # function=evaluar_disenio_batch (abajo, antes: evaluar_disenio).
    # ------------------------------------------------------------
    try:
        shade = SHADE(
            dimension=dimension,
            pop_size=pop_size,
            lower=lower,
            upper=upper,
            function=evaluar_disenio_batch,   # esta función ejecuta n simulaciones en paralelo y devuelve los fitness
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
    finally:
        # Crítico: si algo de lo de arriba lanza una excepción, esto
        # se ejecuta igual -- si no se llama, los workers quedan
        # colgados esperando trabajo para siempre y mpirun no termina.
        stop_workers()

else:
    # ------------------------------------------------------------
    # Procesos worker: NO corren SHADE. Esperan lotes de vectores de
    # parámetros, evalúan los que les tocan con Meep, y devuelven
    # los fitness al maestro. Salen de este bucle cuando el maestro
    # llama a stop_workers().
    # ------------------------------------------------------------
    worker_loop(evaluar_disenio)



