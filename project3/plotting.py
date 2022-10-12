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
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs('eq2425_2022p3_aillet_bonato/project3')

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

df = pd.concat([
    pd.DataFrame(config_list),
    pd.DataFrame(summary_list),
    pd.DataFrame(name_list, columns=['name'])
], axis=1)

# %%
print(df
    [
        [
            'conv_filters',
            'fully_connected_sizes',
            'conv_kernel_sizes',
            'activation',
            'dropout',
            'batch_normalization',
            'batch_size',
            'learning_rate',
            'data_shuffling',
            'optimizer',
            'accuracy',
            'val_accuracy',
        ]
    ]
    .dropna()
    .sort_values(by=['val_accuracy'], ascending=False)
    .rename(
        columns={
            'accuracy': 'recall',
            'val_accuracy': 'val recall',
            'optimizer': 'optim',
            'learning_rate': 'lr',
            'data_shuffling': 'shuffle',
            'batch_normalization': 'batchnorm',
            'fully_connected_sizes': 'fc sizes',
            'batch_size': 'batch',
            'conv_kernel_sizes': 'filter sizes',
            'conv_filters': 'filters',
        }
    )
    .to_latex(
        index=False, 
        formatters={
            'batch': lambda x: f'{x:.0f}',
            'recall': lambda x: f'{x:.3f}',
            'val recall': lambda x: f'{x:.3f}',
            'filter sizes': lambda ks: ', '.join(f'{kx}×{ky}' for kx,ky in ks),
            'activation': lambda a: a.replace('_', ' ').replace('relu', 'ReLU').replace('leaky', 'Leaky'),
        }
    )
    .replace('\\\n', '\\ \hline\n')
    .replace(r'\toprule', '\hline')
    .replace(r'\midrule', '\hline')
    .replace(r'\bottomrule\n', '')
)

# %%
