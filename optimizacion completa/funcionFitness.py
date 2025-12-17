import numpy as np
import pandas as pd

def compute_penalties_from_constraints(freq, S31, S21, deltaA, bands):
    freq = np.asarray(freq)
    S31 = np.asarray(S31)
    S21 = np.asarray(S21)
    dA  = np.asarray(deltaA)
    penalties = np.zeros((5, len(bands)), dtype=float)

    def band_idx(band):
        fmin, fmax = band
        return np.where((freq >= fmin) & (freq <= fmax))[0]

    def frac_true(boolarr):
        if boolarr.size == 0:
            return 0.0
        return float(np.mean(boolarr.astype(float)))

    for j, band in enumerate(bands):
        idx = band_idx(band)
        S31_r = S31[idx]
        S21_r = S21[idx]
        dA_r  = dA[idx]

        ok0 = (S31_r > -3.5) & (S31_r < -2.5)
        penalties[0, j] = 1.0 - frac_true(ok0)

        ok1 = (S21_r > -3.5) & (S21_r < -2.5)
        penalties[1, j] = 1.0 - frac_true(ok1)

        ok2 = (dA_r > 0.0) & (dA_r < 1.0)         
        penalties[2, j] = 1.0 - frac_true(ok2)

        if S31_r.size >= 3:
            left = S31_r[:-2]
            center = S31_r[1:-1]
            right = S31_r[2:]
            is_max = (center > left) & (center > right)
            is_max_full = np.concatenate(([False], is_max, [False]))
            ok3 = is_max_full & (S31_r > -3.5) & (S31_r < -2.5)
        else:
            ok3 = np.zeros_like(S31_r, dtype=bool)
        penalties[3, j] = 1.0 - frac_true(ok3)

        if S21_r.size >= 3:
            left = S21_r[:-2]
            center = S21_r[1:-1]
            right = S21_r[2:]
            is_max = (center > left) & (center > right)
            is_max_full = np.concatenate(([False], is_max, [False]))
            ok4 = is_max_full & (S21_r > -3.5) & (S21_r < -2.5)
        else:
            ok4 = np.zeros_like(S21_r, dtype=bool)
        penalties[4, j] = 1.0 - frac_true(ok4)

    return penalties

def fitness_from_Sparams(freq,
                         S31_dB,
                         S21_dB,
                         deltaA_dB,
                         W=None,
                         Penalties=None,
                         bands=None,
                         lambda_reg=0.008,
                         eps=1e-12,
                         debug=False):
    freq = np.asarray(freq)
    S31 = np.asarray(S31_dB)
    S21 = np.asarray(S21_dB)
    dA  = np.asarray(deltaA_dB)

    if bands is None:
        bands = [(65,70), (71,110), (111,115)]

    if W is None:
        W = np.ones((5,3))
    if Penalties is None:
        Penalties = np.zeros((5,3))
    W = np.asarray(W, dtype=float)
    Penalties = np.asarray(Penalties, dtype=float)

    def indices_in_band(f, band):
        fmin, fmax = band
        return np.where((f >= fmin) & (f <= fmax))[0]

    def mse_to_target(y_region, target):
        if y_region.size == 0:
            return 0.0
        return float(np.mean((y_region - target)**2))

    def osc_intensity(y_region, f_region):
        if y_region.size < 3:
            return 0.0
        dy = np.gradient(y_region, f_region)
        d2y = np.gradient(dy, f_region)
        val = np.abs(np.trapz(d2y, f_region))
        return float(val)

    F = np.zeros((5,3), dtype=float)
    for j, band in enumerate(bands):
        idx = indices_in_band(freq, band)
        f_region = freq[idx]
        S31_r = S31[idx]
        S21_r = S21[idx]
        dA_r  = dA[idx]

        F[0,j] = mse_to_target(S31_r, -3.0)
        F[1,j] = mse_to_target(S21_r, -3.0)
        F[2,j] = mse_to_target(dA_r, 0.0)
        F[3,j] = osc_intensity(S31_r, f_region) if f_region.size>0 else 0.0
        F[4,j] = osc_intensity(S21_r, f_region) if f_region.size>0 else 0.0

    Wpenalized = 100.0 * W + Penalties
    wmax = Wpenalized.max()
    wmin = Wpenalized.min()
    if np.isclose(wmax, wmin):
        Wnorm = np.zeros_like(Wpenalized)
    else:
        Wnorm = (Wpenalized - wmin) / (wmax - wmin)

    product = Wnorm.dot(F.T)
    logscaled = np.log(100.0 * product + eps)
    diag = np.diag(logscaled)
    fitness_value = float(np.sum(diag[0:3]) + lambda_reg * np.sum(diag[3:5]))

    if debug:
        return fitness_value, {
            "F": F,
            "Wpenalized": Wpenalized,
            "Wnorm": Wnorm,
            "product": product,
            "logscaled": logscaled,
            "diag": diag,
            "Penalties": Penalties
        }
    return fitness_value



bands = [(65,70), (71,110), (111,115)]
    


W = np.array([
    [0.005, 0.470, 0.001],
    [0.005, 0.470, 0.001],
    [0.001, 0.005, 0.001],
    [0.001, 0.005, 0.001],
    [0.001, 0.032, 0.001]
], dtype=float)




def evaluar_disenio(x):
    """
    x = [a, b, L=10.0, d, h, c,n=8] mm

    Corre MEEP con esos parámetros,
    calcula S-parámetros y devuelve fitness.
    """

    print("Evaluando diseño con parámetros:", x)
    from geometria_sinpng import ejecutar_simulacion
    freq, S21_dB, S31_dB = ejecutar_simulacion(x)
    
    

    # --------------------------------------------------------------
    # 3) Calcular deltaA 
    # --------------------------------------------------------------
    deltaA_dB = np.abs(S31_dB - S21_dB)

    # --------------------------------------------------------------

    # --------------------------------------------------------------
    Penalties_auto = compute_penalties_from_constraints(
        freq, S31_dB, S21_dB, deltaA_dB, bands
    )

    # --------------------------------------------------------------
  
    # --------------------------------------------------------------
    fitness_val = fitness_from_Sparams(
        freq,
        S31_dB,
        S21_dB,
        deltaA_dB,
        W=W,
        Penalties=Penalties_auto,
        bands=bands,
        debug=False
    )

    return fitness_val








