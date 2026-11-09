# SHADE.py
import csv

import numpy as np
import random
import copy
import sys
import os

# MOdificado por BRayhna para guardar la totalidad de las simulaciones de cada iteración en un archivo CSV

if not os.path.exists("particulas_optimizacion.csv"):
    try:
        # "x" = creación exclusiva: falla si otro proceso ya lo creó.
        # Necesario porque ahora varios procesos MPI importan este
        # módulo casi al mismo tiempo al arrancar (uno por núcleo);
        # sin esto, podrían pisarse unos a otros al crear el archivo.
        with open("particulas_optimizacion.csv", "x", newline="") as f:
            f.write("# Optimización mediante SHADE\n")
            f.write("# Fecha: 2026-07-29\n")
            f.write("# Columnas: x1,x2,x3,x4,x5,fitness\n")
    except FileExistsError:
        pass  # otro proceso lo creó primero entre el chequeo y la escritura

def save_population_to_csv(data1, data2):
    with open("particulas_optimizacion.csv", "a", newline="") as f:
        writer = csv.writer(f)

        # Desempaquetar el arreglo y añadir el número
        writer.writerow([*data1, data2])
# hasta aqui mi intervención 



class ParticleSHADE:
    def __init__(self, dimension, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

        # Solución inicial aleatoria dentro del rango permitido
        self.solution = np.random.uniform(self.lower, self.upper, dimension)
        # Fitness inicial (muy grande → para minimizar)

        self.value = sys.float_info.max

class SHADE:
    def __init__(self, dimension, pop_size, lower, upper, function, maxIter,
                 H=10, p=0.2):

        self.dimension = dimension
        self.pop_size = pop_size
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.function = function
        self.maxIter = maxIter
        


        #Inicialización de memoria adaptativa
        self.H = H
        self.M_F = np.ones(H) * 0.5  # Historical de factores F F: Factor de mutación
        self.M_CR = np.ones(H) * 0.5  #Historial de CR Factor de recombinación
        self.k = 0
        self.p = p

        self.history_best = []
        self.history_mean = []
        self.history_diversity = []


        #Creación de la población inicial
        self.population = [ParticleSHADE(dimension, lower, upper) for _ in range(pop_size)]

        # --- Evaluación PARALELA de la población inicial ---
        # self.function ahora recibe una LISTA de soluciones y devuelve
        # una LISTA de fitness en el MISMO ORDEN (una simulación de Meep
        # independiente por núcleo disponible, ver mpi_parallel_eval.py).
        solutions0 = [p.solution for p in self.population]
        values0 = self.function(solutions0)
        for p, v in zip(self.population, values0):
            p.value = v
            save_population_to_csv(p.solution, p.value)  # Guardar la solución y el valor en el archivo CSV

        print("Población inicial (primeros 5):")
        for idx,p in enumerate(self.population[:5]):
            print(f"  ind {idx}: sol={p.solution} val={p.value}") 

        

        self.best = min(self.population, key=lambda p: p.value)
        self.archive = []

    def _clip(self, vec):
        return np.clip(vec, self.lower, self.upper)

    def run(self, verbose=True, tolerance=1e-6):

        iter_to_min = None
        for iteration in range(self.maxIter):
            S_F = []
            S_CR = []
            delta_f = []

            sorted_idx = np.argsort([p.value for p in self.population])
            num_pbest = max(2, int(self.p * self.pop_size))

            # ------------------------------------------------------------
            # FASE 1 -- Generar TODOS los trial de esta generación
            # (secuencial, barato: son solo operaciones de numpy/random,
            # no simulaciones). OJO: 'self.archive' queda FIJO durante
            # toda la generación en vez de crecer individuo a individuo
            # como antes -- es el único cambio de comportamiento real
            # frente a la versión secuencial, necesario porque ahora los
            # trial se evalúan todos juntos, no uno a uno. Es el ajuste
            # estándar en DE/SHADE paralelo.
            # ------------------------------------------------------------
            trials = []
            trial_meta = []  # (F, CR) de cada individuo, para la memoria adaptativa
            archive_snapshot = self.archive

            for i, p in enumerate(self.population):
                r = random.randrange(self.H)
                F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
                F = min(F, 1)

                CR = np.random.normal(self.M_CR[r], 0.1)
                CR = np.clip(CR, 0, 1)

                pbest_idx = random.choice(sorted_idx[:num_pbest]) #Selección del pbest
                pbest = self.population[pbest_idx].solution

                idxs = list(range(self.pop_size)); idxs.remove(i)
                r1 = random.choice(idxs); x_r1 = self.population[r1].solution

                AplusP = [pp.solution for pp in self.population] + archive_snapshot
                x_r2 = random.choice(AplusP)

                #Mutación SHADE
                xi = p.solution
                mutant = xi + F*(pbest - xi) + F*(x_r1 - x_r2) 
                mutant = self._clip(mutant)

                #Recombinación
                trial = xi.copy()
                j_rand = random.randrange(self.dimension)
                for j in range(self.dimension):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trials.append(trial)
                trial_meta.append((F, CR))

            # ------------------------------------------------------------
            # FASE 2 -- Evaluar TODOS los trial EN PARALELO. Aquí es
            # donde se disparan las simulaciones de Meep simultáneas,
            # una por núcleo disponible.
            # ------------------------------------------------------------
            f_trials = self.function(trials)
            for trial, f_trial in zip(trials, f_trials):
                save_population_to_csv(trial, f_trial)  # Guardar cada trial y su valor en el CSV

            # ------------------------------------------------------------
            # FASE 3 -- Selección y actualización de memoria (secuencial,
            # barato) -- misma lógica que antes.
            # ------------------------------------------------------------
            new_population = []
            for i, p in enumerate(self.population):
                F, CR = trial_meta[i]
                trial = trials[i]
                f_trial = f_trials[i]

                if f_trial < p.value:
                    new_p = ParticleSHADE(self.dimension, self.lower, self.upper)
                    new_p.solution = trial
                    new_p.value = f_trial
                    new_population.append(new_p)

                    S_F.append(F); S_CR.append(CR); delta_f.append(p.value - f_trial)
                    self.archive.append(p.solution.copy())
                else:
                    new_population.append(p)

            if len(self.archive) > self.pop_size:
                self.archive = random.sample(self.archive, self.pop_size)

            #Actualizar memoria adaptativa
            if len(S_F) > 0:
                w = np.array(delta_f)/np.sum(delta_f)
                MF_new = np.sum(w * np.array(S_F)**2) / np.sum(w * np.array(S_F))
                MCR_new = np.sum(w * np.array(S_CR))
                self.M_F[self.k] = MF_new
                self.M_CR[self.k] = MCR_new
                self.k = (self.k + 1) % self.H

            self.population = new_population
            current_best = min(self.population, key=lambda p: p.value)
            if current_best.value < self.best.value:
                self.best = copy.deepcopy(current_best)

            vals = [p.value for p in self.population]
            self.history_best.append(min(vals))
            self.history_mean.append(np.mean(vals))
            diversity = np.mean([np.linalg.norm(p.solution - self.best.solution) for p in self.population])
            self.history_diversity.append(diversity)

            if iter_to_min is None and self.best.value <= tolerance:
                iter_to_min = iteration

           
            if verbose:
                print(f"Iter {iteration}: best={self.history_best[-1]:.6f} | mean={self.history_mean[-1]:.6f}")

        return self.best.solution, self.best.value, self.history_best, self.history_mean, self.history_diversity, iter_to_min


