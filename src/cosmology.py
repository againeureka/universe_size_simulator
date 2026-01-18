from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Constants
C_KM_S = 299792.458
MPC_KM = 3.0856775814913673e19
SEC_PER_GYR = 3600 * 24 * 365.25 * 1e9
KM_PER_LY = 9.4607304725808e12
KM_PER_GLY = KM_PER_LY * 1e9


@dataclass(frozen=True)
class LCDMParams:
    """Minimal ΛCDM parameter bundle (a0=1)."""

    H0_km_s_Mpc: float = 67.4
    Omega_m: float = 0.315
    Omega_r: float = 9e-5
    Omega_L: float = 0.685
    Omega_k: float = 0.0

    def normalized(self) -> "LCDMParams":
        # Normalize so Ωr+Ωm+Ωk+ΩΛ=1 (keep Ωk as provided; scale others proportionally)
        total = self.Omega_r + self.Omega_m + self.Omega_L + self.Omega_k
        if total == 0:
            return self
        if abs(total - 1.0) < 1e-9:
            return self
        # Scale non-curvature components
        nonk = self.Omega_r + self.Omega_m + self.Omega_L
        if nonk <= 0:
            return self
        scale = (1.0 - self.Omega_k) / nonk
        return LCDMParams(
            H0_km_s_Mpc=self.H0_km_s_Mpc,
            Omega_m=self.Omega_m * scale,
            Omega_r=self.Omega_r * scale,
            Omega_L=self.Omega_L * scale,
            Omega_k=self.Omega_k,
        )


def H0_to_sinv(H0_km_s_Mpc: float) -> float:
    """Convert H0 from km/s/Mpc to 1/s."""
    return (H0_km_s_Mpc) / MPC_KM


def E_of_a(a: np.ndarray, p: LCDMParams) -> np.ndarray:
    """Dimensionless expansion rate E(a)=H(a)/H0."""
    a = np.asarray(a)
    return np.sqrt(
        p.Omega_r / a**4 + p.Omega_m / a**3 + p.Omega_k / a**2 + p.Omega_L
    )


def H_of_a(a: np.ndarray, p: LCDMParams) -> np.ndarray:
    """H(a) in 1/s."""
    H0 = H0_to_sinv(p.H0_km_s_Mpc)
    return H0 * E_of_a(a, p)


def integrate_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def age_and_horizons(p: LCDMParams, n: int = 20000, a_min: float = 1e-8) -> Dict[str, float]:
    """Compute age and key horizon scales for a ΛCDM model.

    Returns (in Gyr / Gly):
    - age_Gyr
    - particle_horizon_comoving_Gly (χ0)
    - particle_horizon_proper_now_Gly (D_now)
    - hubble_radius_now_Gly (c/H0)
    """
    p = p.normalized()

    # grid in a; log spacing gives good resolution at early times
    a = np.geomspace(a_min, 1.0, n)
    H = H_of_a(a, p)  # 1/s

    # t0 = ∫ da / (a H(a))
    dt_da = 1.0 / (a * H)
    t0_s = integrate_trapz(a, dt_da)

    # χ0 = ∫ c da / (a^2 H(a))  (comoving)
    dchi_da_km = (C_KM_S) / (a**2 * H)  # km per unit-a
    chi0_km = integrate_trapz(a, dchi_da_km)

    # Convert units
    age_Gyr = t0_s / SEC_PER_GYR
    chi0_Gly = chi0_km / KM_PER_GLY

    # Proper distance now (a0=1)
    Dnow_Gly = chi0_Gly

    # Hubble radius now ~ c/H0
    H0_sinv = H0_to_sinv(p.H0_km_s_Mpc)
    Rh0_km = C_KM_S / H0_sinv
    Rh0_Gly = Rh0_km / KM_PER_GLY

    return {
        "age_Gyr": age_Gyr,
        "particle_horizon_comoving_Gly": chi0_Gly,
        "particle_horizon_proper_now_Gly": Dnow_Gly,
        "hubble_radius_now_Gly": Rh0_Gly,
        "Omega_total": p.Omega_r + p.Omega_m + p.Omega_k + p.Omega_L,
    }


def history_curves(p: LCDMParams, n: int = 3000, a_min: float = 1e-6) -> Dict[str, np.ndarray]:
    """Return arrays for plotting: a, t(a), chi(a), D_now_equiv(a).

    - t(a) is cosmic time since a_min (approx Big Bang) in Gyr
    - chi(a) is comoving distance light can travel from a_min to a in Gly
    - D_now_equiv(a) = a0 * chi(a) (with a0=1) gives 'present-day proper distance' of that comoving radius.

    Note: starting at a_min avoids divergence at a=0.
    """
    p = p.normalized()
    a = np.geomspace(a_min, 1.0, n)
    H = H_of_a(a, p)

    dt_da = 1.0 / (a * H)
    t_s = np.cumsum((dt_da[1:] + dt_da[:-1]) * (a[1:] - a[:-1]) / 2.0)
    t_s = np.insert(t_s, 0, 0.0)

    dchi_da_km = C_KM_S / (a**2 * H)
    chi_km = np.cumsum((dchi_da_km[1:] + dchi_da_km[:-1]) * (a[1:] - a[:-1]) / 2.0)
    chi_km = np.insert(chi_km, 0, 0.0)

    t_Gyr = t_s / SEC_PER_GYR
    chi_Gly = chi_km / KM_PER_GLY
    Dnow_Gly = chi_Gly

    # naive ct converted to Gly, for intuition
    ct_Gly = (C_KM_S * (t_s)) / KM_PER_GLY

    return {
        "a": a,
        "t_Gyr": t_Gyr,
        "chi_Gly": chi_Gly,
        "Dnow_Gly": Dnow_Gly,
        "ct_Gly": ct_Gly,
        "E_of_a": E_of_a(a, p),
    }
