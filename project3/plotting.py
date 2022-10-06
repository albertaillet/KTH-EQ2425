# %% Plot relu and leaky relu functions in two subplots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.3):
    return np.maximum(alpha * x, x)

# %%
def plot_activation(activation, ax, title, color):
    l = 5
    y_offset = 2
    x = np.linspace(-5, 5, 100)
    ax.plot(x, activation(x), color=color)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('activation(x)')
    ax.set_ylim(-l+y_offset, l+y_offset)
    ax.set_xlim(-l, l)
    ax.set_aspect('equal')
    ax.grid()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plot_activation(relu, ax1, 'ReLU', 'b')
plot_activation(leaky_relu, ax2, 'Leaky ReLU', 'r')

# %%
