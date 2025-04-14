import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create the figure and the scatter plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
n = np.arange(0, 21)
initial_omega = 0.5
y = np.sin(initial_omega * n)
sc = ax.scatter(n, y, color='blue')
ax.set_ylim(-1.1, 1.1)
ax.set_xticks(n)
ax.grid(True)
ax.set_title(f"$y[n] = \sin({initial_omega:.2f} \cdot n)$")
ax.set_xlabel("n")
ax.set_ylabel("y[n]")

# Add slider for omega
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
omega_slider = Slider(ax_slider, 'ω', 0.0, 2 * np.pi, valinit=initial_omega, valstep=0.1*np.pi)

# Update function for slider
def update(val):
    omega = omega_slider.val
    y = np.sin(omega * n)
    sc.set_offsets(np.c_[n, y])
    ax.set_title(f"$y[n] = \sin({omega:.2f} \cdot n)$")
    fig.canvas.draw_idle()

omega_slider.on_changed(update)

plt.show()