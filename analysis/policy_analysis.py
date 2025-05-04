from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Frequency
f = 0.05
A = 0.5
omega_sq = (2 * np.pi * f)**2

# Define the dynamics: x1' = x2, x2' = u = -omega^2 * sin(2*pi*f*t)
def dynamics(t, state):
    x1, x2 = state
    u = -A*omega_sq * np.sin(2 * np.pi * f * t)
    dx1 = x2
    dx2 = u
    return [dx1, dx2]

# Time span
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Initial conditions: several, plus special one (1, 0)
initial_conditions = [
    [0, 2*np.pi*A*f],           # special one
]

# Solve and plot
plt.figure(figsize=(8, 6))

for i, x0 in enumerate(initial_conditions):
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    label = f"Initial: ({x0[0]}, {x0[1]})"
    plt.plot(sol.y[0], sol.y[1], label=label, linewidth=2 if x0 == [1, 0] else 1)

# Plot settings
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Trajectories in Phase Space (x1 vs x2)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()