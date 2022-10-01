# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
FOLDER = 'output2'
# %%
df = pd.read_json(f'{FOLDER}/results.json')
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
# %%
