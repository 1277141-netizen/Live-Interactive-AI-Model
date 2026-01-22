import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="International Dental Expansion – Calculus Model", layout="wide")

st.title("🌍📈 International Dental Expansion – Calculus Model")
st.markdown("**By: Racely Ortega**")

st.markdown(
    """
This app applies **Calculus (derivatives & second derivatives)** to compare international
dental practice expansion across countries. It visualizes **growth, costs, and seasonality**
and highlights **critical points and inflection points**.
"""
)

# ----------------------------------
# Symbolic variable
# ----------------------------------
x = sp.symbols("x")

# ----------------------------------
# Sidebar – Mode selection
# ----------------------------------
st.sidebar.header("⚙️ Mode")
comparison_mode = st.sidebar.checkbox("Enable Comparison Mode")

# ----------------------------------
# Sidebar – Countries
# ----------------------------------
st.sidebar.header("🌍 Country Selection")

countries = [
    "Spain 🇪🇸",
    "Italy 🇮🇹",
    "United States 🇺🇸",
    "Mexico 🇲🇽",
    "Japan 🇯🇵",
    "Brazil 🇧🇷"
]

country_1 = st.sidebar.selectbox("Primary country", countries)

country_2 = None
if comparison_mode:
    country_2 = st.sidebar.selectbox("Comparison country", countries, index=1)

# Country parameters
country_params = {
    "Spain 🇪🇸":  {"a": 1.2, "b": 0.45, "c": 3},
    "Italy 🇮🇹":  {"a": 1.0, "b": 0.30, "c": 4.5},
    "United States 🇺🇸": {"a": 1.5, "b": 0.35, "c": 6},
    "Mexico 🇲🇽": {"a": 1.1, "b": 0.50, "c": 2.5},
    "Japan 🇯🇵":  {"a": 0.9, "b": 0.25, "c": 5},
    "Brazil 🇧🇷": {"a": 1.3, "b": 0.55, "c": 3}
}

# ----------------------------------
# Sidebar – Function type
# ----------------------------------
st.sidebar.header("📐 Function Type")

function_type = st.sidebar.selectbox(
    "Choose model",
    [
        "Exponential (Market Growth)",
        "Logarithmic (Diminishing Returns)",
        "Polynomial (Cost Structure)",
        "Trigonometric (Seasonality)"
    ]
)

# ----------------------------------
# Function builder
# ----------------------------------
def build_function(params):
    a, b, c = params["a"], params["b"], params["c"]

    if function_type == "Exponential (Market Growth)":
        return a * sp.exp(b * x) + c
    elif function_type == "Logarithmic (Diminishing Returns)":
        return a * sp.log(b * x) + c
    elif function_type == "Polynomial (Cost Structure)":
        return a * x**2 + b * x + c
    elif function_type == "Trigonometric (Seasonality)":
        return a * sp.sin(b * x) + c

# ----------------------------------
# Plot range
# ----------------------------------
xmin, xmax = st.sidebar.slider("Time range", 0.1, 10.0, (0.1, 5.0))
x_vals = np.linspace(xmin, xmax, 1000)

# ----------------------------------
# Plotting
# ----------------------------------
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 11))

def plot_country(country, linestyle="-"):
    params = country_params[country]
    f = build_function(params)

    f_prime = sp.diff(f, x)
    f_double = sp.diff(f_prime, x)

    f_num = sp.lambdify(x, f, "numpy")
    fp_num = sp.lambdify(x, f_prime, "numpy")
    fpp_num = sp.lambdify(x, f_double, "numpy")

    axes[0].plot(x_vals, f_num(x_vals), linestyle=linestyle, label=country)
    axes[1].plot(x_vals, fp_num(x_vals), linestyle=linestyle, label=country)
    axes[2].plot(x_vals, fpp_num(x_vals), linestyle=linestyle, label=country)

    # Critical points: f'(x) = 0
    try:
        critical_points = sp.solve(f_prime, x)
        for cp in critical_points:
            if cp.is_real and xmin <= float(cp) <= xmax:
                axes[0].axvline(float(cp), linestyle="--", alpha=0.6)
                axes[1].axvline(float(cp), linestyle="--", alpha=0.6)
    except Exception:
        pass

    # Inflection points: f''(x) = 0
    try:
        inflection_points = sp.solve(f_double, x)
        for ip in inflection_points:
            if ip.is_real and xmin <= float(ip) <= xmax:
                axes[0].axvline(float(ip), linestyle=":", alpha=0.7)
                axes[2].axvline(float(ip), linestyle=":", alpha=0.7)
    except Exception:
        pass

# Plot primary country
plot_country(country_1)

# Plot comparison country
if comparison_mode and country_2 != country_1:
    plot_country(country_2, linestyle="--")

# ----------------------------------
# Labels & legend
# ----------------------------------
axes[0].set_ylabel("f(x)")
axes[0].set_title("Function (Revenue / Demand / Cost)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel("f'(x)")
axes[1].set_title("First Derivative (Growth Rate)")
axes[1].grid(True)

axes[2].set_ylabel("f''(x)")
axes[2].set_title("Second Derivative (Concavity)")
axes[2].grid(True)

axes[-1].set_xlabel("Time")
plt.tight_layout()
st.pyplot(fig)

# ----------------------------------
# Explanation
# ----------------------------------
st.subheader("📘 Calculus Interpretation")

st.markdown(
"""
- **Critical points (dashed lines)** show when growth stops increasing or decreasing.
- **Inflection points (dotted lines)** show where acceleration changes.
- Comparing countries reveals which markets grow faster, stabilize earlier,
  or become costly to scale.
"""
)
