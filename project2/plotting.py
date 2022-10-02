# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
FOLDER = 'output2'
# %%
df = pd.read_json(f'{FOLDER}/results.json')
df = df.drop(columns=['top_k_recall'])
# %%
df_depth = df[(df['b'] == 4) & (df['n_components'] == 1)].sort_values('depth')
depths = df_depth['depth']
top_1_recall = df_depth['top_1_recall']
top_5_recall = df_depth['top_5_recall']

plt.plot(depths, top_1_recall, 'x-', label='Top-1 Recall')
plt.plot(depths, top_5_recall, 'x-', label='Top-5 Recall')
plt.ylim(0, 1)
plt.xlabel('Depth')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Recall depending on tree depth for b=4')
plt.grid()
plt.savefig(f'{FOLDER}/depth_vs_recall.png')

# %%
df_b = df[(df['depth'] == 5) & (df['n_components'] == 1)].sort_values('b')
bs = df_b['b']
top_1_recall = df_b['top_1_recall']
top_5_recall = df_b['top_5_recall']

plt.plot(bs, top_1_recall, 'x-', label='Top-1 Recall')
plt.plot(bs, top_5_recall, 'x-', label='Top-5 Recall')
plt.ylim(0, 1)
plt.xlabel('b')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Recall depending on b for depth=5')
plt.grid()
plt.savefig(f'{FOLDER}/b_vs_recall.png')
# %% plot each recall vs b for each depth value

df_3d = (df[
    (df['n_components'] == 1) & 
    (df['perc']==1) & 
    (df['b'] >= 3) & 
    (df['depth'] >= 4)]
    .drop_duplicates()
    .sort_values(['b', 'depth'])
)
n_bs = len(df_3d['b'].unique())
n_depths = len(df_3d['depth'].unique())
depths = df_3d['depth'].values.reshape(n_bs, n_depths)
bs = df_3d['b'].values.reshape(n_bs, n_depths)
top_1_recall = df_3d['top_1_recall'].values.reshape(n_bs, n_depths)
top_5_recall = df_3d['top_5_recall'].values.reshape(n_bs, n_depths)

b_colors = plt.get_cmap('autumn')(np.linspace(0.8, 0, n_bs))
depth_colors = plt.get_cmap('summer')(np.linspace(0.8, 0, n_depths))

# %%
fig = plt.figure()
for b, depth, top_1_recall_i, top_5_recall_i, color in reversed(list(zip(bs, depths, top_1_recall, top_5_recall, b_colors))):
    plt.plot(depth, top_1_recall_i, 'x-', label=f'Top-1 Recall for b={b[0]}', color=color)
plt.ylim(0, 1)
plt.xlabel('Depth')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Top 1 Recall depending on tree depth for different values of b')
plt.grid()
plt.savefig(f'{FOLDER}/depth_vs_top1recall.png')

# %%
fig = plt.figure()
for b, depth, top_1_recall_i, top_5_recall_i, color in reversed(list(zip(bs, depths, top_1_recall, top_5_recall, b_colors))):
    plt.plot(depth, top_5_recall_i, 'x-', label=f'Top-5 Recall for b={b[0]}', color=color)
plt.ylim(0, 1)
plt.xlabel('Depth')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Top 5 Recall depending on tree depth for different values of b')
plt.grid()
plt.savefig(f'{FOLDER}/depth_vs_top5recall.png')

# %%
fig = plt.figure()
for b, depth, top_1_recall_i, top_5_recall_i, color in reversed(list(zip(bs.T, depths.T, top_1_recall.T, top_5_recall.T, depth_colors))):
    plt.plot(b, top_1_recall_i, 'x-', label=f'Top-1 Recall for depth={depth[0]}', color=color)
plt.ylim(0, 1)
plt.xlabel('b')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Recall depending on b for different depth values')
plt.xticks(bs[:, 0])
plt.grid()
plt.savefig(f'{FOLDER}/b_vs_top1recall.png')

# %%
fig = plt.figure()
for b, depth, top_1_recall_i, top_5_recall_i, color in reversed(list(zip(bs.T, depths.T, top_1_recall.T, top_5_recall.T, depth_colors))):
    plt.plot(b, top_5_recall_i, 'x-', label=f'Top-5 Recall for depth={depth[0]}', color=color)
plt.ylim(0, 1)
plt.xlabel('b')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.title('Recall depending on b for different depth values')
plt.xticks(bs[:, 0])
plt.grid()
plt.savefig(f'{FOLDER}/b_vs_top5recall.png')

# %%
