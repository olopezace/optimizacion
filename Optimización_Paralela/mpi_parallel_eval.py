"""
Capa de paralelización MPI para evaluar en paralelo N vectores de
parámetros independientes (una simulación de Meep por vector).

Patrón maestro-trabajador:
  - El proceso rank 0 (maestro) corre TODA la lógica de SHADE de forma
    normal y secuencial -- igual que en la versión sin paralelizar.
  - Los demás procesos (workers) NO corren SHADE. Se quedan esperando
    lotes de vectores de parámetros, evalúan SOLO los que les tocan
    (repartidos round-robin entre todos los procesos, incluido rank 0),
    y devuelven sus resultados al maestro.

Cada proceso queda aislado como una simulación de Meep independiente
(divide_parallel_processes), así que dentro de cada proceso
ejecutar_simulacion() corre exactamente igual que en la versión serial
-- no necesita saber que está corriendo en paralelo.

Ejecutar SIEMPRE con (el -m mpi4py evita que un proceso caído deje
colgados a los demás):
    mpirun -np N python -m mpi4py main.py
"""

from mpi4py import MPI
import meep as mp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Aísla cada proceso como una simulación de Meep independiente (sin
# esto, Meep asumiría que los N procesos deben cooperar en UNA sola
# simulación por descomposición de dominio). Debe llamarse UNA sola
# vez, antes de crear cualquier Simulation -- por eso va aquí, a nivel
# de módulo, para que se ejecute automáticamente al importar.
mp.divide_parallel_processes(mp.count_processors())


def _evaluar_mi_parte(vectores, worker_fn):
    """Cada proceso evalúa vectores[rank::size]. Si worker_fn falla para
    algún vector (p.ej. una geometría inválida que revienta con un
    IndexError), se captura el error y se penaliza con fitness=inf en
    vez de tumbar todo el trabajo MPI."""
    mis_resultados = {}
    for i in range(rank, len(vectores), size):
        try:
            mis_resultados[i] = worker_fn(vectores[i])
        except Exception as e:
            print(f"[rank {rank}] fallo evaluando vector {i}: {e}", flush=True)
            mis_resultados[i] = float("inf")
    return mis_resultados


def evaluate_batch(vectores, worker_fn):
    """Llamar SOLO desde rank 0 (el maestro). Reparte 'vectores' entre
    todos los procesos disponibles, corre worker_fn(v) en cada uno en
    paralelo, y devuelve la lista de resultados en el MISMO ORDEN que
    'vectores'."""
    assert rank == 0, "evaluate_batch solo se debe llamar desde el proceso maestro (rank 0)"
    comm.bcast(("WORK", vectores), root=0)
    parciales = _evaluar_mi_parte(vectores, worker_fn)
    todos = comm.gather(parciales, root=0)
    fusionado = {}
    for d in todos:
        fusionado.update(d)
    return [fusionado[i] for i in range(len(vectores))]


def worker_loop(worker_fn):
    """Llamar SOLO desde los procesos que NO son el maestro (rank != 0).
    Se queda esperando lotes de trabajo hasta recibir la señal de
    parada enviada por stop_workers()."""
    assert rank != 0, "worker_loop no se debe llamar desde el proceso maestro"
    while True:
        msg = comm.bcast(None, root=0)
        if msg[0] == "STOP":
            break
        _, vectores = msg
        parciales = _evaluar_mi_parte(vectores, worker_fn)
        comm.gather(parciales, root=0)


def stop_workers():
    """Llamar SOLO desde rank 0, al terminar toda la optimización, para
    que los procesos worker puedan salir de su bucle de espera. Sin
    esto, los workers quedan colgados esperando un lote que nunca
    llega y mpirun nunca termina."""
    assert rank == 0
    comm.bcast(("STOP", None), root=0)
