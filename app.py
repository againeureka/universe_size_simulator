from __future__ import annotations

import streamlit as st

from src.cosmology import LCDMParams, age_and_horizons, history_curves as lcdm_history
from src.plotting import plot_horizon, plot_rate_vs_a, plot_scale_factor
from src.toy import ToyParams, history_curves as toy_history, summary as toy_summary


def fmt(x: float, unit: str = "") -> str:
    return f"{x:,.3g}{(' ' + unit) if unit else ''}"


@st.cache_data(show_spinner=False)
def compute_toy(tp: ToyParams):
    return toy_summary(tp), toy_history(tp)


@st.cache_data(show_spinner=False)
def compute_lcdm(p: LCDMParams):
    return age_and_horizons(p), lcdm_history(p)


st.set_page_config(page_title="Expanding Universe Simulator", layout="wide")

st.title("Expanding Universe Simulator")
st.caption(
    "Toy model (ruler intuition) and LambdaCDM (FRW). The key: age t0 and the present-day particle-horizon radius D_now need not match."
)

st.sidebar.header("Controls")
model = st.sidebar.radio("Model", ["Toy (ruler intuition)", "LambdaCDM (FRW)"], index=0)


if model.startswith("Toy"):
    st.sidebar.subheader("Toy parameters")
    mode = st.sidebar.selectbox(
        "Scale factor mode",
        ["power_law", "exponential", "piecewise_inflation"],
        index=2,
    )
    age = st.sidebar.slider("Age t0 (Gyr)", 0.1, 30.0, 13.8, 0.1)
    p_exp = st.sidebar.slider("p (power-law exponent)", 0.1, 2.0, 0.67, 0.01)
    H = st.sidebar.slider("H (Gyr^-1) for exponential", 0.001, 3.0, 0.07, 0.001)
    t_end = st.sidebar.number_input(
        "Inflation end time (Gyr)",
        min_value=1e-12,
        max_value=1.0,
        value=1e-6,
        format="%.2e",
    )
    N = st.sidebar.slider("N e-folds (piecewise)", 0.0, 100.0, 50.0, 1.0)

    tp = ToyParams(
        mode=mode,
        age_Gyr=age,
        p=p_exp,
        H_Gyr_inv=H,
        t_infl_end_Gyr=t_end,
        N_efolds=N,
    )
    summ, curves = compute_toy(tp)

    c1, c2, c3 = st.columns(3)
    c1.metric("Age t0", fmt(summ["age_Gyr"], "Gyr"))
    c2.metric("Observable radius D_now", fmt(summ["particle_horizon_proper_now_Gly"], "Gly"))
    c3.metric("D_now / (c*t0)", fmt(summ["particle_horizon_proper_now_Gly"] / float(curves["ct_Gly"][-1])))

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_scale_factor(curves["t_Gyr"], curves["a"], title="Toy: a(t)"))
    with right:
        st.pyplot(
            plot_horizon(
                curves["t_Gyr"],
                curves["Dnow_Gly"],
                curves["ct_Gly"],
                title="Toy: D_now(t) vs naive c*t",
            )
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Tip: Increase N e-folds in piecewise_inflation to see how an early exponential phase makes D_now much larger than c*t0."
    )

    st.markdown(
        """
### Intuition (Toy)
Space is a ruler whose tick spacing grows like a(t).
- Light always moves locally at speed c.
- In tick units, light advances by c/a(t) per unit time.
- Early times have tiny a(t), so light crosses many ticks.
- Later expansion stretches those ticks, so the *present-day* distance can exceed c*t0.
"""
    )

else:
    st.sidebar.subheader("LambdaCDM parameters")
    H0 = st.sidebar.slider("H0 (km/s/Mpc)", 50.0, 90.0, 67.4, 0.1)
    Om = st.sidebar.slider("Omega_m", 0.0, 1.0, 0.315, 0.001)
    Or = st.sidebar.slider("Omega_r", 0.0, 0.01, 9e-5, 1e-5, format="%.5f")
    Ok = st.sidebar.slider("Omega_k", -0.2, 0.2, 0.0, 0.001)
    OL = st.sidebar.slider("Omega_Lambda", 0.0, 1.5, 0.685, 0.001)

    params = LCDMParams(H0_km_s_Mpc=H0, Omega_m=Om, Omega_r=Or, Omega_k=Ok, Omega_L=OL)
    summ, curves = compute_lcdm(params)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age t0", fmt(summ["age_Gyr"], "Gyr"))
    c2.metric("Observable radius D_now", fmt(summ["particle_horizon_proper_now_Gly"], "Gly"))
    c3.metric("Hubble radius c/H0", fmt(summ["hubble_radius_now_Gly"], "Gly"))
    c4.metric("D_now / (c*t0)", fmt(summ["particle_horizon_proper_now_Gly"] / float(curves["ct_Gly"][-1])))

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_scale_factor(curves["t_Gyr"], curves["a"], title="LambdaCDM: a(t)"))
        st.pyplot(plot_rate_vs_a(curves["a"], curves["E_of_a"], title="LambdaCDM: E(a)=H(a)/H0"))
    with right:
        st.pyplot(
            plot_horizon(
                curves["t_Gyr"],
                curves["Dnow_Gly"],
                curves["ct_Gly"],
                title="LambdaCDM: D_now(t) vs naive c*t",
            )
        )

    st.markdown(
        """
### Intuition (FRW)
Particle horizon uses the comoving light-travel distance: chi = integral c dt / a(t).
Early times have small a(t), so dt contributes a lot of comoving distance.
That comoving scale is later stretched, so the present-day particle-horizon radius D_now can exceed c*t0.
"""
    )

st.divider()
st.subheader("Definitions")
st.markdown(
    """
- Age t0: cosmic proper time since the early universe.
- Observable radius D_now (particle horizon): the farthest we can see today, expressed as a present-day proper distance.
- naive c*t: comparison curve for a static ruler.
"""
)
