from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_scale_factor(t_Gyr: np.ndarray, a: np.ndarray, title: str = "Scale factor"):
    fig, ax = plt.subplots()
    ax.plot(t_Gyr, a)
    ax.set_xlabel("Cosmic time t (Gyr)")
    ax.set_ylabel("a(t) (a(t0)=1)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    return fig


def plot_horizon(t_Gyr: np.ndarray, Dnow_Gly: np.ndarray, ct_Gly: np.ndarray, title: str = "Horizon vs naive c·t"):
    fig, ax = plt.subplots()
    ax.plot(t_Gyr, Dnow_Gly, label="Present-day proper distance of particle horizon")
    ax.plot(t_Gyr, ct_Gly, label="Naive c·t")
    ax.set_xlabel("Cosmic time t (Gyr)")
    ax.set_ylabel("Distance (Gly)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
    return fig


def plot_rate_vs_a(a: np.ndarray, E: np.ndarray, title: str = "E(a)=H(a)/H0"):
    fig, ax = plt.subplots()
    ax.plot(a, E)
    ax.set_xlabel("Scale factor a")
    ax.set_ylabel("E(a)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True)
    return fig
