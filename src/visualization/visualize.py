import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt


# plot the ground truth trajectory and predicted trajectory for individuals
def plot_truth_against_pred(df_results, group=None, cogmeasure='PZGLOBAL', alpha_scaler=2):
    n = df_results.shape[0]
    if group!=None:
        n_groups = len(df_results[group].unique())
        colors = plt.cm.Accent(np.linspace(0,1,n_groups))
        groups = list(df_results[group].unique())
        groups.sort()
        mapping = dict(zip(groups, range(n_groups)))
        color = df_results[group].apply(lambda x: colors[mapping[x]])
    else:
        color = plt.cm.Accent(np.linspace(0,1,n))
    cnt = 0
    for idx,row in df_results.iterrows():
        col_truth = [x for x in df_results.columns if '_truth' in x]
        col_pred = [x for x in df_results.columns if '_pred' in x]
        age = [0] + [float(x.split('_')[1].split('yr')[0]) for x in row[col_truth].dropna().index.tolist()]
        cog_truth = [row[cogmeasure]] + row[col_truth].dropna().tolist()
        cog_pred = [row[cogmeasure]] + row[col_pred].dropna().tolist()
        if len(age)==1:
            plt.scatter(age, cog_truth, c=np.expand_dims(color[cnt], axis=0), s=3)
            plt.scatter(age, cog_pred, c=np.expand_dims(color[cnt], axis=0), marker='+')
        elif len(age)>1:
            alpha = np.clip(np.abs(np.tanh(cog_truth[-1] - cog_pred[-1]))*alpha_scaler, a_min=0, a_max=1)
            plt.plot(age, cog_truth, c=color[cnt], alpha=alpha)
            plt.plot(age, cog_pred, c=color[cnt], linestyle='--', alpha=alpha)
        cnt += 1
    return