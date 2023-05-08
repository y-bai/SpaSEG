import random
import os
import torch
import scanpy as sc
import numpy as np
import anndata as ad

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_max_H_W(adata_list):
    H = 0
    W = 0
    for adata in adata_list:
        col, row = adata.obs['array_col'].values.astype(int), adata.obs['array_row'].values.astype(int)
        H = max(W, row.max()+1)
        W = max(H, col.max()+1)
    return H, W

def get_3d_expMatrix(adata, channel, H, W):
    #x_pca_scale = sc.pp.scale(adata.obsm['X_pca'])
    x_pca_scale = sc.pp.scale(adata.obsm['X_pca'], copy=True)
    col = adata.obs['array_col'].values.astype(int)
    row = adata.obs['array_row'].values.astype(int)
    poss = zip(col, row)
    # build and fill the 3D matrix of each spot with the corresponding PCA
    # mxt = np.zeros((channel, max(row) + 1, max(col) + 1))
    mxt = np.zeros((channel, H, W))

    for i, idx in enumerate(poss):
        mxt[:, idx[1], idx[0]] = x_pca_scale[i, :]
    return mxt, col, row

def add_embedding(adata, H_embedding, W_embedding, embedding, opt):
    SpaSEG_embedding = embedding.reshape((H_embedding, W_embedding, opt.nChannel))
    col = adata.obs['array_col'].values.astype(int)
    row = adata.obs['array_row'].values.astype(int)
    shape = adata.obsm['X_pca'].shape

    poss = zip(col, row)
    SpaSEG_pca = np.zeros(shape)

    for i, idx in enumerate(poss):
        SpaSEG_pca[i, :] = SpaSEG_embedding[idx[1], idx[0], :]

    adata.obsm["SpaSEG_embedding"] = SpaSEG_pca

    return adata


def outlier(arrayMatrix):
    arraystd = np.std(arrayMatrix)
    arraymean = np.mean(arrayMatrix)
    arrayoutlier = np.where(np.abs(arrayMatrix - arraymean) > (arraystd))  # or 2*arraystd)
    # arrayoutlier=np.transpose(np.where(np.abs(arrayMatrix-arraymean)>(arraystd)))#or 2*arraystd)
    return arrayoutlier


def merge_outlier(target, data, im, device):
    #####################obtain cutoff#####################
    nLabels = len(np.unique(np.array(target.cpu())))
    labels = np.array(target.cpu())
    labels = labels.reshape(im.shape[0] * im.shape[1] * im.shape[2])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    l_avg_a = []
    for i in range(len(u_labels)):
        dataxx = data.permute(0, 2, 3, 1).contiguous().view(-1, data.shape[1])
        xxx = dataxx[l_inds[i], :]

        l_avg = np.mean(xxx.cpu().detach().numpy(), axis=0)
        l_avg_a.append(l_avg.reshape(-1))
    dist = []
    dist_ind_i = []
    dist_ind_j = []
    for i in range(len(u_labels)):
        for j in range(len(u_labels)):
            if j > i:
                dist.append(np.linalg.norm(l_avg_a[i] - l_avg_a[j]))
                dist_ind_i.append(i)
                dist_ind_j.append(j)
    output_outlier = outlier(np.array(dist))

    idx = np.where(dist == np.min(dist))[0][0]
    # print (output_outlier,'yyyyyyy')
    if output_outlier[0] != [] and idx in output_outlier[0]:
        index_need_change = np.where(labels == u_labels[dist_ind_j[idx]])
        target[index_need_change] = torch.as_tensor(u_labels[dist_ind_i[idx]]).to(device)
    else:
        target = target

    return target

def cal_metric(adata, ground_truth_index):
    if ground_truth_index:
        adata.obs['ground_truth_code'] = adata.obs[ground_truth_index].cat.codes
        ground_truth = adata.obs['ground_truth_code']
        pred_labels = adata.obs['SpaSEG_clusters']
        # nmi
        NMI = normalized_mutual_info_score(np.array(ground_truth), np.array(pred_labels))
        print('nmi=', NMI, end='      ')
        # ari
        ARI = adjusted_rand_score(np.array(ground_truth), np.array(pred_labels))
        print('ari=', ARI)
        metric_dict = {"ARI": ARI, "NMI": NMI}

        adata.uns["metrics"] = metric_dict
    else:
        input_feature_X = adata.obsm['X_pca']
        pred_labels = adata.obs['SpaSEG_clusters']
        CHS = calinski_harabasz_score(input_feature_X, pred_labels)
        SC = silhouette_score(input_feature_X, pred_labels)
        DBS = davies_bouldin_score(input_feature_X, pred_labels)
        metric_dict = {"CHS": CHS, "SC": SC, "DBS": DBS}
        adata.uns["metrics"] = metric_dict

    return adata

def batch_umap_plot(adata_list, sample_id_list):
    adata_map = {sample_id:adata for sample_id, adata in zip(sample_id_list, adata_list)}
    adatas = ad.concat(adata_map, join="inner", index_unique="_", label="batch")
    #adatas.obs['SpaSEG_batch_clusters'] = adatas.obs['SpaSEG_clusters'].astype('str')

    # visualize UMAP before batch correction using embedding from PCA
    sc.pp.neighbors(adatas, use_rep="X_pca", key_added="neighbor_X_pca")
    sc.tl.umap(adatas, neighbors_key="neighbor_X_pca")
    sc.pl.umap(adatas, color="batch", neighbors_key="neighbor_X_pca", title="Uncorrected",
                      save='SpaSEG_Uncorrected_batch.pdf', show=False, frameon=False)
    sc.pl.umap(adatas, color="SpaSEG_clusters", neighbors_key="neighbor_X_pca", title="Uncorrected",
                      save='SpaSEG_Uncorrected_clusters.pdf', show=False, frameon=False)

    # visualize UMAP after batch correction using embedding from SpaSEG
    sc.pp.neighbors(adatas, use_rep="SpaSEG_embedding", key_added="neighbor_SpaSEG")
    sc.tl.umap(adatas, neighbors_key="neighbor_SpaSEG")
    sc.pl.umap(adatas, color="batch", neighbors_key="neighbor_SpaSEG", title="SpaSEG",
                      save='SpaSEG_corrected_batch.pdf', show=False, frameon=False)
    sc.pl.umap(adatas, color="SpaSEG_clusters", neighbors_key="neighbor_SpaSEG", title="SpaSEG",
                      save='SpaSEG_corrected_clusters.pdf', show=False, frameon=False)