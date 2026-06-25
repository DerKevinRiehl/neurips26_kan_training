"""Quick visual inspection for Feynman functions.

Run this file directly in Spyder.
Comment/uncomment ``FUNCTION_NAME`` lines below to inspect one function at a time.
For functions with more than two inputs, the first two variables are plotted and
the remaining variables are fixed to the midpoint of their domains.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from feynman_db import get_function, list_functions
except ImportError:
    from src.experiment_1_feynman.feynman_db import get_function, list_functions


# Pick one function by uncommenting or editing this line.
# FUNCTION_NAME = "I.16.6"
# FUNCTION_NAME = "I.6.2"
# FUNCTION_NAME = "I.6.2b"
# FUNCTION_NAME = "I.9.18"
# FUNCTION_NAME = "I.12.11"
# FUNCTION_NAME = "I.13.12"
# FUNCTION_NAME = "I.15.3x"
FUNCTION_NAME = "I.18.4"
# FUNCTION_NAME = "I.26.2"
# FUNCTION_NAME = "I.27.6"
# FUNCTION_NAME = "I.29.16"
# FUNCTION_NAME = "I.30.3"
# FUNCTION_NAME = "I.30.5"
# FUNCTION_NAME = "I.37.4"
# FUNCTION_NAME = "I.40.1"
# FUNCTION_NAME = "I.44.4"
# FUNCTION_NAME = "I.50.26"
# FUNCTION_NAME = "II.2.42"
# FUNCTION_NAME = "II.6.15a"
# FUNCTION_NAME = "II.11.7"
# FUNCTION_NAME = "II.11.27"
# FUNCTION_NAME = "II.35.18"
# FUNCTION_NAME = "II.36.38"
# FUNCTION_NAME = "II.38.3"
# FUNCTION_NAME = "III.9.52"
# FUNCTION_NAME = "III.10.19"
# FUNCTION_NAME = "III.17.37"

N_GRID = 200
SHOW_3D_SURFACE = True
PRINT_AVAILABLE_FUNCTIONS = False


if PRINT_AVAILABLE_FUNCTIONS:
    for function in list_functions():
        print(
            function.name,
            "inputs=",
            function.variables,
            "domains=",
            function.domains,
            "expr=",
            function.expression,
        )


function = get_function(FUNCTION_NAME)
print(f"{function.name}: {function.expression}")
print(f"variables: {function.variables}")
print(f"domains: {function.domains}")

lows = np.array([domain[0] for domain in function.domains], dtype=float)
highs = np.array([domain[1] for domain in function.domains], dtype=float)
midpoint = (lows + highs) / 2.0

if function.input_dim == 1:
    x = np.linspace(lows[0], highs[0], N_GRID)
    X = x.reshape(-1, 1)
    y = function.evaluate(X)[:, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel(function.variables[0])
    plt.ylabel("f(x)")
    plt.title(f"{function.name}: {function.expression}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

else:
    x0 = np.linspace(lows[0], highs[0], N_GRID)
    x1 = np.linspace(lows[1], highs[1], N_GRID)
    X0, X1 = np.meshgrid(x0, x1)
    X = np.tile(midpoint, (N_GRID * N_GRID, 1))
    X[:, 0] = X0.ravel()
    X[:, 1] = X1.ravel()
    y = function.evaluate(X)[:, 0].reshape(N_GRID, N_GRID)

    finite = np.isfinite(y)
    if not np.all(finite):
        y = np.where(finite, y, np.nan)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X0, X1, y, levels=50, cmap="viridis")
    plt.colorbar(contour, label="f(x)")
    plt.xlabel(function.variables[0])
    plt.ylabel(function.variables[1])
    plt.title(f"{function.name}: {function.expression}")
    if function.input_dim > 2:
        fixed = ", ".join(
            f"{name}={value:.3g}"
            for name, value in zip(function.variables[2:], midpoint[2:])
        )
        plt.suptitle(f"Holding {fixed}", y=0.02, fontsize=9)
    plt.tight_layout()

    if SHOW_3D_SURFACE:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X0, X1, y, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel(function.variables[0])
        ax.set_ylabel(function.variables[1])
        ax.set_zlabel("f(x)")
        ax.set_title(f"{function.name}: {function.expression}")
        plt.tight_layout()

plt.show()
