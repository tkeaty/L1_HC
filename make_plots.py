import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from math import fabs


def transform_dist_mat(mat, labels):
    max_val = np.max(mat)
    inds = []

    for i in range(mat.shape[0]):
        if i == 0:
            l_min = max_val
        else:
            l_min = np.min(mat[i, :i])

        if i < mat.shape[0]-1:
            r_min = np.min(mat[i, i+1:])
        else:
            r_min = max_val

        if l_min < max_val and r_min < max_val:
            inds.append(i)

    mat = mat[inds, :]
    mat = mat[:, inds]
    new_labels = labels[inds]

    dist = None
    for i in range(mat.shape[0]):
        if dist is None:
            dist = mat[i, i+1:]
        else:
            dist = np.append(dist, mat[i, i+1:])

    return dist, new_labels


all = pd.read_csv('Outputs/Paths/ALL_gene_set.csv', header=None).values
aml = pd.read_csv('Outputs/Paths/AML_gene_set.csv', header=None).values
ctrl = pd.read_csv('Outputs/Paths/Ctrl_gene_set.csv', header=None).values

with open('gene_set.csv', 'r') as f:
    for line in f:
        g = line.split(',')[:-1]

genes = np.asarray(g, dtype=object)

# ALL plot
all_dist, all_genes = transform_dist_mat(all, genes)
all_link = linkage(all_dist, optimal_ordering=True)

plt.figure(figsize=(8, 12))
plt.title('ALL Network - Curated functional genes')
plt.ylabel('Gene Name')
plt.xlabel('Cluster Distance')
dendrogram(all_link, orientation='left', labels=all_genes, leaf_rotation=45, color_threshold=0)
plt.savefig('Outputs/Figures/ALL_gene_set.png')
plt.close()

# AML plot
aml_dist, aml_genes = transform_dist_mat(aml, genes)
aml_link = linkage(aml_dist, optimal_ordering=True)

plt.figure(figsize=(8, 12))
plt.title('AML Network - Curated functional genes')
plt.ylabel('Gene Name')
plt.xlabel('Cluster Distance')
dendrogram(aml_link, orientation='right', labels=aml_genes, leaf_rotation=45, color_threshold=0)
plt.savefig('Outputs/Figures/AML_gene_set.png')
plt.close()

# Ctrl plot
ctrl_dist, ctrl_genes = transform_dist_mat(ctrl, genes)
ctrl_link = linkage(ctrl_dist, optimal_ordering=True)

plt.figure(figsize=(12, 8))
plt.title('Control Network - Curated functional genes')
plt.xlabel('Gene Name')
plt.ylabel('Cluster Distance')
dendrogram(all_link, orientation='right', labels=all_genes, leaf_rotation=45, color_threshold=0)
plt.savefig('Outputs/Figures/Ctrl_gene_set.png')
plt.close()
