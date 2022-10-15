# %%
import pandas as pd 
import wandb
api = wandb.Api()

# %% Project is specified by <entity/project-name>
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
sweep4AB = set([
    'ethereal-sweep-8',
    'avid-sweep-7',
    'unique-sweep-6',
    'pious-sweep-5',
    'gallant-sweep-4',
    'proud-sweep-3',
    'devoted-sweep-2',
    'morning-sweep-1',
])
sweep4CDE = set([
    'eager-sweep-8',
    'robust-sweep-7',
    'tough-sweep-6',
    'solar-sweep-5',
    'generous-sweep-4',
    'crimson-sweep-3',
    'divine-sweep-2',
    'dauntless-sweep-1',
])
sweep5ABC = set([
    'pleasant-sweep-8',
    'morning-sweep-7',
    'different-sweep-6',
    'rosy-sweep-5',
    'royal-sweep-4',
    'misunderstood-sweep-3',
    'earnest-sweep-2',
    'ethereal-sweep-1',
])
sweep_aug_lr_decay = set([
    'worthy-sweep-4',
    'royal-sweep-3',
    'summer-sweep-2',
    'dark-sweep-1',
])
sweep_optimizers = set([
    'light-sweep-3',
    'serene-sweep-2',
    'glad-sweep-1',
])

# %%
df['val recall'] = df['best_val_accuracy'].fillna(df['val_accuracy'])
print(df
    [
        df['name'].isin(sweep_optimizers)
    ]
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
            'val recall',
        ]
    ]
    .dropna()
    .sort_values(by=['val recall'], ascending=False)
    .rename(
        columns={
            'accuracy': 'recall',
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
    .replace(r'\bottomrule', '')
)
# %%
df['val recall'] = df['best_val_accuracy'].fillna(df['val_accuracy'])
print(df
    [
        df['name'].isin(sweep_aug_lr_decay)
    ]
    [
        [
            'exponential_decay',
            'augmentation',
            'exp_dec_init_lr',
            'decay_rate',
            'decay_steps',
            'accuracy',
            'val recall',
        ]
    ]
    .dropna()
    .sort_values(by=['val recall'], ascending=False)
    .rename(
        columns={
            'accuracy': 'recall',
            'exponential_decay': 'exp dec',
            'augmentation': 'aug',
            'exp_dec_init_lr': 'init lr',
            'decay_rate': 'decay rate',
            'decay_steps': 'decay steps',
        }
    )
    .to_latex(
        index=False, 
        formatters={
            'batch': lambda x: f'{x:.0f}',
            'recall': lambda x: f'{x:.3f}',
            'val recall': lambda x: f'{x:.3f}',
            'decay rate': lambda x: f'{x:.1f}',
            'decay steps': lambda x: f'{x:.0f}',
        }
    )
    .replace('\\\n', '\\ \hline\n')
    .replace(r'\toprule', '\hline')
    .replace(r'\midrule', '\hline')
    .replace(r'\bottomrule', '')
)

# %%
