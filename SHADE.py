# SHADE.py
import numpy as np
import random
import copy
import sys

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
        for p in self.population:
            p.value = self.function(p.solution)

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

            #Generación de nuevos individuos
            new_population = []
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

                AplusP = [pp.solution for pp in self.population] + self.archive
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

                #Selección
                f_trial = self.function(trial)

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


