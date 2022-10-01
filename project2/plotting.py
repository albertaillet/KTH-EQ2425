# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
# %% plot 3d plane with b, depth, recall

df_3d = df[(df['n_components'] == 1) & (df['perc']==1)].drop_duplicates().sort_values(['b', 'depth'])
n_bs = len(df_3d['b'].unique())
n_depths = len(df_3d['depth'].unique())
depths = df_3d['depth'].values.reshape(n_bs, n_depths)
bs = df_3d['b'].values.reshape(n_bs, n_depths)
top_1_recall = df_3d['top_1_recall'].values.reshape(n_bs, n_depths)
top_5_recall = df_3d['top_5_recall'].values.reshape(n_bs, n_depths)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(bs, depths, top_1_recall, label='Top-1 Recall', alpha=0.5)
ax.plot_surface(bs, depths, top_5_recall, label='Top-5 Recall', alpha=0.5)
ax.set_xlabel('b')
ax.set_ylabel('depth')
ax.set_zlabel('Recall')
plt.title('Recall depending on b and depth')

# %%
