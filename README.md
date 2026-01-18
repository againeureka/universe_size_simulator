# Expanding Universe Simulator (Streamlit)

A small Streamlit app to build intuition for why the **age of the universe** (t0) and the **present-day radius of the observable universe** (particle horizon, D_now) do **not** have to match in an expanding universe.

## What's inside

- **Toy model (ruler intuition)**
  - Space = a ruler whose tick spacing grows with a scale factor a(t)
  - Light always moves locally at speed c
  - Comoving distance is the *number of ticks* crossed: chi = ∫ c dt / a(t)
  - Present-day horizon distance can exceed c * t0

- **LambdaCDM (FRW)**
  - H(a) = H0 * sqrt(Or/a^4 + Om/a^3 + Ok/a^2 + OL)
  - Age: t0 = ∫ da / (a H(a))
  - Particle horizon: chi0 = ∫ c da / (a^2 H(a))

## Run

Create an environment with Streamlit and run:

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

## Files

- `app.py` : Streamlit UI
- `src/cosmology.py` : minimal LambdaCDM integrals (no astropy)
- `src/toy.py` : toy scale-factor models + comoving horizon integral
- `src/plotting.py` : matplotlib plots used by Streamlit
