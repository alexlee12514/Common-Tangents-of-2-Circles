import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, I

# Function to find real common tangents
def find_common_tangents_real(h1, k1, r1, h2, k2, r2):
    # Define variables to solve for
    m, b = symbols('m b')
    
    # Define the system of equations
    eq1 = Eq((2*b*m - 2*k1*m - 2*h1)**2 - 4*(1 + m**2)*(h1**2 + (b - k1)**2 - r1**2), 0)
    eq2 = Eq((2*b*m - 2*k2*m - 2*h2)**2 - 4*(1 + m**2)*(h2**2 + (b - k2)**2 - r2**2), 0)
    
    # Solve symbolically
    solutions = solve((eq1, eq2), (m, b), dict=True)

    # Filter out complex solutions
    real_solutions = [sol for sol in solutions if not sol[m].has(I) and not sol[b].has(I)]
    print("Non-vertical tangents: ", real_solutions)
    
    return real_solutions

# Function to find vertical tangents
def find_vertical_solutions(h1, k1, r1, h2, k2, r2):
    vertical_solutions = set()  # Use a set to store unique values
    
    if h1 - r1 == h2 - r2:
        vertical_solutions.add(h1 - r1)
    if h1 + r1 == h2 + r2:
        vertical_solutions.add(h1 + r1)
    if h1 + r1 == h2 - r2:
        vertical_solutions.add(h1 + r1)
    if h1 - r1 == h2 + r2:
        vertical_solutions.add(h1 - r1)

    print("Vertical tangents: ", vertical_solutions)
    return vertical_solutions  # A set automatically ensures uniqueness


# Function to plot circles and tangents with equal scaling
def plot_common_tangents(h1, k1, r1, h2, k2, r2):
    # Solve for real common tangents
    solutions = find_common_tangents_real(h1, k1, r1, h2, k2, r2)
    vertical_solutions = find_vertical_solutions(h1, k1, r1, h2, k2, r2)  # Get vertical tangents

    # Generate the circles
    theta = np.linspace(0, 2 * np.pi, 300)
    circle1_x = h1 + r1 * np.cos(theta)
    circle1_y = k1 + r1 * np.sin(theta)
    circle2_x = h2 + r2 * np.cos(theta)
    circle2_y = k2 + r2 * np.sin(theta)

    # Plot the circles
    plt.figure(figsize=(8, 8))
    plt.plot(circle1_x, circle1_y, 'b', label="Circle 1")
    plt.plot(circle2_x, circle2_y, 'g', label="Circle 2")
    plt.scatter([h1, h2], [k1, k2], color='black', marker='o', label="Centers")

    # Plot the sloped tangents
    x_vals = np.linspace(min(h1, h2) - r1 - r2, max(h1, h2) + r1 + r2, 100)

    for sol in solutions:
        m_val = sol[symbols('m')].evalf()
        b_val = sol[symbols('b')].evalf()

        if m_val.is_finite and b_val.is_finite:
            if m_val == 0:  # Handle horizontal tangent
                plt.axhline(y=b_val, color='r', linestyle='--', label=f"Tangent: y={b_val:.2f}")
            else:
                tangent_y = m_val * x_vals + b_val
                plt.plot(x_vals, tangent_y, 'r--', label=f"Tangent: y={m_val:.2f}x + {b_val:.2f}")

    # Plot the vertical tangents
    for vt in vertical_solutions:
        plt.axvline(x=vt, color='r', linestyle='--', label=f"Vertical Tangent at x={vt:.2f}")

    # Labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title("Common Tangents to Two Circles")
    plt.grid()

    # Adjust axis scaling to be equal
    plt.axis('equal')  # Ensures equal scaling for x and y

    plt.show()

# Example usage
plot_common_tangents(1, 2, 3, 4, 5, 6)
