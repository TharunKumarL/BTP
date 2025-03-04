import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import h5py
from cycler import cycler
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import seaborn as sns
import pandas as pd

def read_h5py(path):
    with h5py.File(path, 'r') as hf:
        return np.array(hf['dataset'][:])

def return_clustering(mat, k=None):
    if k is None:
        k = mat.shape[1]
    
    cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='complete')
    clusts = cl.fit_predict(mat)
    
    uniq = np.unique(clusts)
    _map = dict(zip(uniq, range(len(uniq))))
    lbls = np.array([_map[i] for i in clusts])
    return lbls


def random_test(N, k, groundtruth, val, repeats=50000):
    vals = []
    for _ in range(repeats):
        v = np.random.choice(np.arange(k), N, replace=True)
        score = adjusted_rand_score(groundtruth, v)
        vals.append(score)
    vals = np.array(vals)
    return (1 + np.sum(vals > val)) / repeats

def f1_score(G_dis, G_pat):
    G_pat /= np.linalg.norm(G_pat, axis=1, keepdims=True)
    G_dis /= np.linalg.norm(G_dis, axis=1, keepdims=True)
    diagnosis = read_h5py('../matrices/d2p.h5py')
    pr = G_dis @ G_pat.T
    pred = np.argmax(pr, axis=0)
    true = np.argmax(diagnosis, axis=0)
    cm = confusion_matrix(true, pred)
    pr = np.diag(cm) / np.maximum(1, np.sum(cm, 1))
    rc = np.diag(cm) / np.maximum(1, np.sum(cm, 0))
    return np.mean(2 * pr * rc / (pr + rc + 1e-8))

# Load embeddings
G_pat = read_h5py('../preprocesseddata/G_patients.h5py')
G_dis = read_h5py('../preprocesseddata/G_diseases.h5py')
n = G_dis.shape[0]
print(n)
G = np.vstack((G_pat, G_dis))

# Perform t-SNE visualization
embed = TSNE(metric='cosine', perplexity=50)
out = embed.fit_transform(G)
pat = out[:-n]
dis = out[-n:]

# Clustering and Evaluation
lbls = return_clustering(G_pat, k=21)
cmap = np.loadtxt('../dataset/diseases.txt').astype(int)
tissues = np.loadtxt('../dataset/donor_tissue.txt').astype(int)

scores = []
pvals = []
types = ['ARI Cancers', 'ARI Tissues']
v = adjusted_rand_score(cmap, lbls)
p = random_test(G_pat.shape[0], 21, cmap, v)
scores.append(v)
pvals.append(p)

lbls = return_clustering(G_pat, k=np.max(tissues))
v = adjusted_rand_score(tissues, lbls)
p = random_test(G_pat.shape[0], np.max(tissues), cmap, v)
scores.append(v)
pvals.append(p)
print(len(types), len(scores))
min_len = min(len(types), len(scores))
df = pd.DataFrame({'x': types[:min_len], 'scores': scores[:min_len]})
df.to_csv('scores.tsv', sep='\t')

fig, ax = plt.subplots(figsize=(9, 12))
sns.set(font_scale=2)
for i, g in enumerate(np.unique(cmap)):
    ix = np.where(cmap == g)[0]
    ax.scatter(pat[ix, 0], pat[ix, 1], s=3*(matplotlib.rcParams['lines.markersize'] ** 2), edgecolors='white')
plt.axis('off')
plt.savefig('../results/embedding_disease_lbls.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 12))
for i, g in enumerate(np.unique(lbls)):
    ix = np.where(lbls == g)[0]
    ax.scatter(pat[ix, 0], pat[ix, 1], edgecolors='white', s=3*(matplotlib.rcParams['lines.markersize'] ** 2), label=f'cluster {int(g)}')
plt.axis('off')
plt.savefig('../results/embedding_cluster_lbls.pdf', bbox_inches='tight')
plt.close()

min_len = min(len(types), len(scores))
types = types[:min_len]
scores = scores[:min_len]

