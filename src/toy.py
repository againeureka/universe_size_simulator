from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from .cosmology import C_KM_S, KM_PER_GLY, SEC_PER_GYR

ToyMode = Literal["power_law", "exponential", "piecewise_inflation"]


@dataclass(frozen=True)
class ToyParams:
    mode: ToyMode = "power_law"
    age_Gyr: float = 13.8

    # power_law: a(t)=(t/t0)^p
    p: float = 2.0 / 3.0

    # exponential: a(t)=exp(H*t)/exp(H*t0) so that a(t0)=1
    H_Gyr_inv: float = 0.07

    # piecewise_inflation: early exp then power law
    t_infl_end_Gyr: float = 1e-6
    N_efolds: float = 50.0


def scale_factor(t_Gyr: np.ndarray, tp: ToyParams) -> np.ndarray:
    t = np.asarray(t_Gyr)
    t0 = tp.age_Gyr
    # avoid zero
    t_safe = np.maximum(t, 1e-18)

    if tp.mode == "power_law":
        return (t_safe / t0) ** tp.p

    if tp.mode == "exponential":
        # a(t)=exp(H t) / exp(H t0)
        H = tp.H_Gyr_inv
        return np.exp(H * t_safe) / np.exp(H * t0)

    if tp.mode == "piecewise_inflation":
        # Inflation: exponential until t_end with N efolds total, then power-law to t0
        t_end = max(tp.t_infl_end_Gyr, 1e-18)
        # define H_infl so that exp(H_infl*t_end)=exp(N)
        H_infl = tp.N_efolds / t_end

        a_end = np.exp(H_infl * t_end)
        # after t_end, match onto power law a(t)=A*(t/t_end)^p
        # choose A so a(t_end)=a_end
        A = a_end
        a_after = A * (np.maximum(t_safe, t_end) / t_end) ** tp.p

        a_raw = np.where(t_safe <= t_end, np.exp(H_infl * t_safe), a_after)
        # normalize so a(t0)=1
        a0 = np.where(t0 <= t_end, np.exp(H_infl * t0), A * (t0 / t_end) ** tp.p)
        return a_raw / a0

    raise ValueError(f"Unknown toy mode: {tp.mode}")


def history_curves(tp: ToyParams, n: int = 4000) -> Dict[str, np.ndarray]:
    """Compute toy history curves.

    Returns arrays:
    - t_Gyr, a(t)
    - chi_Gly: comoving horizon (light-travel 'tick count') up to t
    - Dnow_Gly: present-day proper distance equivalent (=chi for a0=1)
    - ct_Gly: naive c*t
    """
    t0 = tp.age_Gyr

    # log spacing captures early era
    t = np.geomspace(1e-12, t0, n)
    a = scale_factor(t, tp)

    # chi = ∫ c dt / a(t)
    dt_s = (t[1:] - t[:-1]) * SEC_PER_GYR
    integrand = C_KM_S / a  # km/s divided by dimensionless a => km/s
    # trapezoid in km: c dt / a
    dchi_km = (integrand[1:] + integrand[:-1]) / 2.0 * dt_s
    chi_km = np.insert(np.cumsum(dchi_km), 0, 0.0)

    chi_Gly = chi_km / KM_PER_GLY
    Dnow_Gly = chi_Gly  # with a0=1

    ct_km = C_KM_S * (t * SEC_PER_GYR)
    ct_Gly = ct_km / KM_PER_GLY

    return {
        "t_Gyr": t,
        "a": a,
        "chi_Gly": chi_Gly,
        "Dnow_Gly": Dnow_Gly,
        "ct_Gly": ct_Gly,
    }


def summary(tp: ToyParams) -> Dict[str, float]:
    curves = history_curves(tp, n=6000)
    age_Gyr = tp.age_Gyr
    Dnow_Gly = float(curves["Dnow_Gly"][-1])
    return {
        "age_Gyr": age_Gyr,
        "particle_horizon_proper_now_Gly": Dnow_Gly,
        "particle_horizon_comoving_Gly": Dnow_Gly,
    }
