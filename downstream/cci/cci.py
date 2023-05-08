# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2022/10/19 15:42
@License: (C) Copyright 2013-2022.
@File: cci_analysis.py
@Desc:
'''
from typing import Union

import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from scipy.stats import entropy
from anndata import AnnData
import time

from numba import njit, prange
from numba.typed import List
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests


def cal_spot_lr_score(
        adata: AnnData,
        domain_label: str = 'SpaSEG_clusters',
        lr_pvalue: Union[pd.DataFrame, None] = None,
        cell_types_label: str = None,
        min_cell_type_frac: float = 0.3,
        min_expr: float = 0.0,
        min_spot: int = 1,
        n_perm_pairs: int = 400,
        n_rings: int = 2,
        coord_type: str = 'grid',
        n_neighs: int = 10,
        specise: str = 'mouse',
):
    '''

    calculate LR score for each spot.

    Parameters
    ----------
    adata: AnnData
        gene expression
    domain_label:
        domains identifier, saved in adata.obs.
    lr_pvalue:
        storing LR pvalue calculated by CellPhoneDB.
        format: LR * (domain1|domain1, domain1|domain2,...).
        if `None`, then call CellPhoneDB
    cell_types_label: str
        fraction of annotated cell types, it would be saved in  adata.obsm.
        format: spot * (cell_tpye1, cell_tpye2, ...).
        Note: for each spot, sum(axis=1) = 1.
    min_cell_type_frac:
        minimum fraction of cell type in a spot to calculate cell type density
        value would be 1/(number of cell types)
    min_expr:
        used for performing permutation test
    min_spot:
        minimum number of spots having score for the given LR pair
    n_perm_pairs:
        number of gene pairs used for permutation test,
        recommended: n_perm_pairs ~= n_lr_pairs ** 2
    n_rings:
        the number of rings to obtain neighbors for given spot
    coord_type:
        will be `grid` or `generic`, see `sq.gr.spatial_neighbors`
    n_neighs:
        the number of neighbors for given spot
    specise:
        indicator the specise of mouse or human

    Returns
    -------

    '''

    # calculate neighbors
    sq.gr.spatial_neighbors(
        adata, n_rings=n_rings, coord_type=coord_type, n_neighs=n_neighs
    )

    n_spots = len(adata.obs_names)

    neigh_ls = []
    neigh_ix = []
    for i in range(n_spots):
        _, idx = adata.obsp['spatial_distances'][i, :].nonzero()
        neigh_ls.append(','.join(adata.obs_names[idx]))
        neigh_ix.append(idx)

    adata.obs['spot_neighbors'] = pd.DataFrame(neigh_ls, index=adata.obs_names, columns=['spot_neighbors'])

    # get ligand and receptor
    if not isinstance(lr_pvalue, pd.DataFrame) or lr_pvalue is None:
        if domain_label not in adata.obs.keys():
            raise ValueError(
                'Could not find the key {0} in adata.obs'.format(domain_label)
            )

        res = sq.gr.ligrec(
            adata,
            n_perms=1000,
            cluster_key=domain_label,
            copy=True,
            use_raw=False,
            corr_method='fdr_bh',
            corr_axis='clusters',
            alpha=0.05,
            interactions_params={'resources': 'CellPhoneDB'},
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"},
            n_jobs=1,
        )

        adata.uns['domain_lr_means'] = res['means']
        adata.uns['domain_lr_pvalues'] = res['pvalues']
        adata.uns['domain_lr_metadata'] = res['metadata']
        lr_pvalue = res['pvalues']

    pvalue_lr = lr_pvalue.copy()
    pvalue_lr.columns = ['|'.join(col) for col in pvalue_lr.columns.values]
    if specise == 'mouse':
        pvalue_lr.index = [
            col[0].capitalize() + '|' + col[1].capitalize()
            for col in pvalue_lr.index.values
        ]
    else:
        pvalue_lr.index = [
            col[0].upper() + '|' + col[1].upper()
            for col in pvalue_lr.index.values
        ]
    

    # check if the spot has the lr, based on the domain
    start = time.perf_counter()
    spot_lr = pd.DataFrame(index=adata.obs_names)
    for i in range(pvalue_lr.shape[0]):  # l|r x domain1|domain2
        i_domains = [x.split('|') for x in pvalue_lr.columns[pvalue_lr.iloc[i] < 0.05].values]
        if len(i_domains) > 0:
            i_lr_uniq_domains = np.unique(np.array(i_domains).ravel())
            spot_lr_ = adata.obs[domain_label].astype(str).isin(i_lr_uniq_domains)
            spot_lr_.name = pvalue_lr.index[i]
            spot_lr = pd.concat((spot_lr, spot_lr_), axis=1)

    adata_df = adata.to_df()
    adata_df.columns = [x.capitalize() if specise == 'mouse' else x.upper() for x in adata_df.columns]

    # filter lr
    filter_lr_ls = [
        x for x in spot_lr.columns
        if x.split('|')[0] in adata_df.columns and x.split('|')[1] in adata_df.columns
    ]

    spot_lr = spot_lr[filter_lr_ls]
    # save into adata
    # adata.obsm['spot_has_lr'] = spot_lr
    end = time.perf_counter()
    print('Elapsed when finding spot having lr pairs {}s'.format(end - start))

    n_filtered_lr_pairs = len(filter_lr_ls)
    start = time.perf_counter()
    _spot_lr_expr, spot_rl_expr = _get_gene_expr(adata_df, filter_lr_ls)
    _spot_has_lr_split = np.full((n_spots, n_filtered_lr_pairs * 2), False)
    # _spot_lr_expr = np.zeros((n_spots, n_filtered_lr_pairs * 2))
    # lr_split_genes = []
    for i, x in enumerate(filter_lr_ls):
        # l1|r1, l2|r1
        _spot_has_lr_split[:, i * 2] = spot_lr[x]  # ligand
        _spot_has_lr_split[:, i * 2 + 1] = spot_lr[x]  # receptor

    spot_lr_expr = _spot_lr_expr * _spot_has_lr_split

    end = time.perf_counter()
    print('Elapsed when obtaining gene expression for spots and their neighbors {}s'.format(end - start))

    # estimate cell type fraction for each spots
    spot_hetero = np.ones(len(adata.obs_names))
    if cell_types_label is not None:
        ct_frac = adata.obsm[cell_types_label].values  # 0-1 range
        spot_hetero = entropy(ct_frac, base=2, axis=1)
        spot_hetero[np.isnan(spot_hetero)]=0

    start = time.perf_counter()
    lr_score = _cal_score(                                      
        n_spots,
        np.arange(n_spots),
        n_filtered_lr_pairs,
        neigh_ix,
        min_expr,
        spot_hetero,
        spot_lr_expr,
        spot_rl_expr
    )
    lr_has = (lr_score > 0).sum(axis=0) > min_spot
    lr_pair_final = np.array(filter_lr_ls)[lr_has]
    lr_scores = lr_score[:, lr_has]
    end = time.perf_counter()
    print('Elapsed when calculating lr score for spots {}s'.format(end - start))

    adata.uns['lr_pairs'] = lr_pair_final
    adata.uns['neigh_ix'] = neigh_ix
    adata.uns['spot_hetero'] = spot_hetero
    adata.obsm['spot_lr_score'] = pd.DataFrame(
        lr_scores,
        columns=lr_pair_final,
        index=adata.obs_names
    )

    permut_test(
        adata,
        neigh_ix,
        min_expr,
        spot_hetero,
        n_pairs=n_perm_pairs,
        specise=specise
    )
    


def _cal_score(
    n_spots,
    spot_idx,
    n_filtered_lr_pairs,
    neigh_ix,
    min_expr,
    spot_hetero,
    spot_lr_expr,
    spot_rl_expr
):

    lr_score = np.zeros((n_spots, n_filtered_lr_pairs))
    lr_score_split = _cal_score_split(
        spot_idx,
        neigh_ix,
        min_expr,
        spot_hetero,
        spot_lr_expr,
        spot_rl_expr
    )

    _cal_lr_score(lr_score, lr_score_split)                  

    return lr_score


def _cal_score_split(
    spot_idx,
    neigh_ix,
    min_expr,
    spot_hetero,
    spot_lr_expr,
    spot_rl_expr

):
    """

    Calculate score of ligand and receptor separately for each spot

    Parameters
    ----------
    spot_idx
    neigh_ix
    min_expr
    spot_hetero
    spot_lr_expr
    spot_rl_expr

    Returns
    -------

    """
    lr_score_split = np.zeros((len(spot_idx), spot_lr_expr.shape[1]))
    for j, ix in enumerate(spot_idx):
        # get neighbor spots
        i_spot_nbs_idx = neigh_ix[ix]

        i_spot_lr_expr_r = spot_lr_expr[ix, :]
        i_spot_lr_expr = i_spot_lr_expr_r * (i_spot_lr_expr_r > min_expr)

        i_nbs_rl_expr_r = spot_rl_expr[i_spot_nbs_idx, :]
        i_nbs_rl_expr = i_nbs_rl_expr_r * (i_nbs_rl_expr_r > min_expr)

        i_lr_sqrt = np.sqrt(i_spot_lr_expr * i_nbs_rl_expr)
        if len(i_lr_sqrt) > 0:
            lr_score_split[j, :] = i_lr_sqrt.mean(axis=0) * spot_hetero[ix]

    return lr_score_split


@njit(parallel=True)
def _cal_lr_score(lr_score, lr_score_split):
    """

    calculate the lr pair score

    Parameters
    ----------
    lr_score
    lr_score_split

    Returns
    -------

    """
    for i in prange(lr_score_split.shape[1] // 2):
#     for i in range(len(lr_score_split) // 2):
        lr_score[:, i] = (lr_score_split[:, i * 2] + lr_score_split[:, i * 2 + 1]) / 2


def _get_lr_bg(
    lr_pair,
    lr_score: np.ndarray,
    l_q,
    r_q,
    neigh_ix,
    min_expr,
    spot_hetero,
    bg_genes,
    excl_qs,
    excl_genes,
    n_genes,
    n_pairs,
    gene_expr: pd.DataFrame,
):
    """

    Gets the LR-specific background & bg spot indices.

    Parameters
    ----------
    lr_pair:
         a given real lr_pair
    lr_score:
        corresponding LR score for all spots
    l_q:
        quantiles for the L
    r_q:
        quantiles for the R
    neigh_ix
    min_expr
    spot_hetero
    bg_genes
    excl_qs
    excl_genes
    n_genes
    n_pairs
    gene_expr

    Returns
    -------

    """

    # real LR pairs
    l_, r_ = lr_pair.split('|')
    # saving background genes
    if l_ not in bg_genes:
        l_genes = _get_similar_genes(l_q, n_genes, excl_qs, excl_genes)
        bg_genes[l_] = l_genes
    else:
        l_genes = bg_genes[l_]

    if r_ not in bg_genes:
        r_genes = _get_similar_genes(r_q, n_genes, excl_qs, excl_genes)
        bg_genes[r_] = r_genes
    else:
        r_genes = bg_genes[r_]

    random_pair = _gen_rand_pairs(l_genes, r_genes, n_pairs)
    spot_idx = np.where(lr_score > 0)[0]

    spot_lr_expr, spot_rl_expr = _get_gene_expr(gene_expr, random_pair)

    bg_pair_score = _cal_score(
        len(spot_idx),
        spot_idx,
        len(random_pair),
        neigh_ix,
        min_expr,
        spot_hetero,
        spot_lr_expr,
        spot_rl_expr
    )

    return bg_pair_score, spot_idx


def _get_gene_expr(
    gene_expr: pd.DataFrame,
    lr_pairs
):

    n_pairs = len(lr_pairs)
    n_spots = gene_expr.shape[0]
    _spot_lr_expr = np.zeros((n_spots, n_pairs * 2))
    _spot_rl_expr = np.zeros((n_spots, n_pairs * 2))
    # lr_split_genes = []
    for i, x in enumerate(lr_pairs):
        lr_genes = x.split('|')
        # l1|r1, l2|r1
        _spot_lr_expr[:, i * 2] = gene_expr[lr_genes[0]].values
        _spot_lr_expr[:, i * 2 + 1] = gene_expr[lr_genes[1]].values

        _spot_rl_expr[:, i * 2] = gene_expr[lr_genes[1]].values
        _spot_rl_expr[:, i * 2 + 1] = gene_expr[lr_genes[0]].values

        # lr_split_genes.extend(lr_genes)

    return _spot_lr_expr, _spot_rl_expr


@njit
def _gen_rand_pairs(
    genes1: np.array,
    genes2: np.array,
    n_pairs: int
):
    """

    Generates random pairs of genes.
    """

    rand_pairs = List()
    for j in range(0, n_pairs):
        l_rand = np.random.choice(genes1, 1)[0]
        r_rand = np.random.choice(genes2, 1)[0]
        rand_pair = "|".join([l_rand, r_rand])
        while rand_pair in rand_pairs or l_rand == r_rand:
            l_rand = np.random.choice(genes1, 1)[0]
            r_rand = np.random.choice(genes2, 1)[0]
            rand_pair = "|".join([l_rand, r_rand])

        rand_pairs.append(rand_pair)

    return rand_pairs


@njit(parallel=True)
def _get_similar_genes(
    ref_qs: np.array,
    n_genes: int,
    c_qs: np.ndarray,
    c_genes: np.array,
):
    """



    Parameters
    ----------
    ref_qs:
        reference quantiles, such as l_quantiles or r_quantiles
    n_genes:
        the number of similar genes to be returned
    c_qs:
        candidate quantiles
    c_genes

    Returns
    -------

    """

    # Measuring distances from the desired gene #
    dists = np.zeros((1, c_qs.shape[1]), dtype=np.float64)[0, :]
    for i in prange(0, c_qs.shape[1]):
        i_c_qs = c_qs[:, i]
        abs_diff = ref_qs - i_c_qs
        abs_diff[abs_diff < 0] = -abs_diff[abs_diff < 0]
        dists[i] = np.nansum(abs_diff / (ref_qs + i_c_qs))

    # Need to remove the zero-dists since this indicates they are expressed
    # exactly the same, & hence likely in the same spot !
    nonzero_bool = dists > 0
    dists = dists[nonzero_bool]
    c_qs = c_qs[:, nonzero_bool]
    c_genes = c_genes[nonzero_bool]
    order = np.argsort(dists)

    # Retrieving desired number of genes #
    similar_genes = c_genes[order[0:n_genes]]

    return similar_genes


def _get_lr_feature(
        lr_pairs,
        lrs_split_genes: np.ndarray,
        lr_uniq_gene_expr: pd.DataFrame,
        quantiles
):
    """

    get expression features of LR pairs: nonzero-median, zero-prop, quantiles

    """

    # get quantiles of gene expression
    l_gene_idx = []
    r_gene_idx = []
    for i in range(len(lrs_split_genes) // 2):
        l_gene_idx.extend(
            np.where(lr_uniq_gene_expr.columns.values == lrs_split_genes[i * 2])[0]
        )
        r_gene_idx.extend(
            np.where(lr_uniq_gene_expr.columns.values == lrs_split_genes[i * 2 + 1])[0]
        )

    lr_q, l_q, r_q = _get_lr_quantile(
        lr_uniq_gene_expr,
        l_gene_idx,
        r_gene_idx,
        quantiles,
        'quantiles')

    # Calculating the zero proportions,
    # for grouping based on median/zeros
    lr_props, l_props, r_props = _get_lr_zeroprops(
        lr_uniq_gene_expr, l_gene_idx, r_gene_idx)

    # Getting lr features for later diagnostics
    # The nonzero median when quantiles=.5
    # lr_meds: n_lr_pairs x 2
    lr_meds, l_meds, r_meds = _get_lr_quantile(
        lr_uniq_gene_expr,
        l_gene_idx,
        r_gene_idx,
        quantiles=np.array([0.5]),
        method=""
    )
    lr_median_means = lr_meds.mean(axis=1)
    lr_prop_means = lr_props.mean(axis=1)

    # Calculating mean rank #
    median_order = np.argsort(lr_median_means)
    prop_order = np.argsort(lr_prop_means * -1)
    median_ranks = [np.where(median_order == i)[0][0] for i in range(len(lr_pairs))]
    prop_ranks = [np.where(prop_order == i)[0][0] for i in range(len(lr_pairs))]
    mean_ranks = np.array([median_ranks, prop_ranks]).mean(axis=0)

    # Saving the lr features...
    cols = ["nonzero-median", "zero-prop", "median_rank", "prop_rank", "mean_rank"]
    lr_features = pd.DataFrame(index=lr_pairs, columns=cols)
    lr_features.iloc[:, 0] = lr_median_means
    lr_features.iloc[:, 1] = lr_prop_means
    lr_features.iloc[:, 2] = np.array(median_ranks)
    lr_features.iloc[:, 3] = np.array(prop_ranks)
    lr_features.iloc[:, 4] = np.array(mean_ranks)
    lr_features = lr_features.iloc[np.argsort(mean_ranks), :]
    lr_cols = [f"L_{quant}" for quant in quantiles] + [
        f"R_{quant}" for quant in quantiles
    ]

    quant_df = pd.DataFrame(lr_q, columns=lr_cols, index=lr_pairs)
    lr_features = pd.concat((lr_features, quant_df), axis=1)
    return lr_features


def _get_lr_zeroprops(
    lr_expr: pd.DataFrame,
    l_idx: list,
    r_idx: list
):
    """Gets the proportion of zeros per gene in the LR pair, & then concatenates.
    Returns
    -------
    lr_props, l_props, r_props: np.ndarray
    First is concatenation of two latter. Each row is a prop value, each column is a LR pair.
    """

    # First getting the quantiles of gene expression #
    gene_props = np.apply_along_axis(_getzero_prop, 0, lr_expr.values)

    l_props = gene_props[:, l_idx]
    r_props = gene_props[:, r_idx]

    lr_props = np.concatenate((l_props, r_props), 0).transpose()

    return lr_props, l_props, r_props


def _getzero_prop(expr):
    """Calculating the proportion of zeros."""
    zero_bool = expr == 0
    n_zeros = len(np.where(zero_bool)[0])
    zero_prop = [n_zeros / len(expr)]
    return zero_prop


def _get_lr_quantile(
    lr_expr: pd.DataFrame,
    l_idx,
    r_idx,
    quantiles,
    method,
):
    """

    calculate quantiles of LR expression


    """
    q_func = _nonzero_quantile if method != "quantiles" else np.quantile

    # First getting the quantiles of gene expression #
    lr_expr_qs = np.apply_along_axis(
        q_func, 0, lr_expr.values, q=quantiles, interpolation="nearest"
    )

    # quantile for ligand
    # n_quantiles X n_lr_pairs
    l_q = lr_expr_qs[:, l_idx]
    # quantile for receptor
    # n_quantiles X n_lr_pairs
    r_q = lr_expr_qs[:, r_idx]

    # n_lr_pairs x (n_quantiles * 2)
    lr_q = np.concatenate((l_q, r_q), 0).transpose()
    return lr_q, l_q, r_q


def _nonzero_quantile(
    expr,
    q,
    interpolation
):
    """

    Calculating the non-zero quantiles.

    """

    # get all gene expression > 0
    nonzero_expr = expr[expr > 0]
    qts = np.quantile(nonzero_expr, q=q, interpolation=interpolation)
    if not isinstance(qts, np.ndarray):
        qts = np.array([qts])
    return qts


def permut_test(
    adata: AnnData,
    neigh_ix,
    min_expr,
    spot_hetero,
    n_pairs: int = 1000,
    specise: str = 'mouse',
    adj_method: str = "fdr_bh"
):
    """

    permutation test for LR pairs.
    reference: stLearn.

    Parameters
    ----------
    adata
    neigh_ix
    min_expr
    spot_hetero
    n_pairs
    specise
    adj_method

    Returns
    -------

    """
    # get gene names in lr and not in lr.
    if 'lr_pairs' not in adata.uns_keys():
        raise ValueError(f"No 'lr_pairs' found in adata.uns_keys()")

    gene_expr = adata.to_df()
    re_cols = np.array([x.capitalize() if specise == 'mouse' else x.upper() for x in gene_expr.columns])
    gene_expr.columns = re_cols
    genes = re_cols

    lr_pairs = adata.uns['lr_pairs']
    lrs_split_genes = []
    for x in lr_pairs:
        lrs_split_genes.extend(x.split('|'))

    lr_uniq_genes = np.unique(lrs_split_genes)
    excl_genes = np.array([x for x in genes if x not in lr_uniq_genes])

    n_genes = round(np.sqrt(n_pairs) * 2)
    if len(excl_genes) < n_genes:
        raise ValueError(f"too large {n_pairs} to generate lr pairs, recommend 1000")

    # get gene expressions
    lr_uniq_gene_expr = gene_expr[lr_uniq_genes]
    excl_gene_expr = gene_expr[excl_genes]

    # get features for lrs
    start = time.perf_counter()
    quantiels = np.array([0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 1])
    lr_features = _get_lr_feature(
        lr_pairs,
        lrs_split_genes,
        lr_uniq_gene_expr,
        quantiels
    )
    adata.uns['lr_features'] = lr_features

    # quantiles to select similar gene to LRs
    l_qs = lr_features.loc[
        lr_pairs, [col for col in lr_features.columns if 'L_' in col]
    ].values
    r_qs = lr_features.loc[
        lr_pairs, [col for col in lr_features.columns if 'R_' in col]
    ].values

    excl_qs = np.apply_along_axis(
        np.quantile, 0, excl_gene_expr, q=quantiels, interpolation="nearest"
    )
    end = time.perf_counter()
    print('Elapsed when generating features for permutation test {}s'.format(end - start))

    # Ensuring consistent typing to prevent numba errors #
    l_qs = l_qs.astype("<f4")
    r_qs = r_qs.astype("<f4")
    excl_qs = excl_qs.astype("<f4")

    lr_scores = adata.obsm['spot_lr_score'].values
    pvals = np.ones(lr_scores.shape, dtype=np.float64)
    pvals_adj = np.ones(lr_scores.shape, dtype=np.float64)

    with tqdm(
        total=lr_scores.shape[1],
        desc="Generating gene pairs and permutation testing each LR pair...",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        disable=False,
    ) as pbar:
        # Keep track of genes which can be used to generate random pairs.
        bg_genes = {}
        spot_lr_indices = [
            [] for i in range(lr_scores.shape[0])
        ]  # tracks the lrs tested in a given spot for MHT !!!!

        for lr_j in range(lr_scores.shape[1]):
            lr_ = lr_pairs[lr_j]

            start = time.perf_counter()
            bg_pair_score, spot_idx = _get_lr_bg(
                lr_,
                lr_scores[:, lr_j],
                l_qs[lr_j, :],
                r_qs[lr_j, :],
                neigh_ix,
                min_expr,
                spot_hetero,
                bg_genes,
                excl_qs,
                excl_genes,
                n_genes,
                n_pairs,
                gene_expr,
            )

            # Calculate empirical p-values per-spot
            for spot_i, spot_index in enumerate(spot_idx):
                n_greater = len(
                    np.where(bg_pair_score[spot_i, :] >= lr_scores[spot_index, lr_j])[0]
                )

                n_greater = n_greater if n_greater != 0 else 1  # pseudocount
                pvals[spot_index, lr_j] = n_greater / bg_pair_score.shape[1]
                spot_lr_indices[spot_index].append(lr_j)

            end = time.perf_counter()
            print('Permutation test for {}: {}s'.format(lr_, end - start))

            pbar.update(1)

        start = time.perf_counter()
        for spot_i in range(lr_scores.shape[0]):
            lr_indices = spot_lr_indices[spot_i]
            if len(lr_indices) != 0:
                pvals_adj[spot_i, lr_indices] = multipletests(
                    pvals[spot_i, lr_indices], method=adj_method
                )[1]
        end = time.perf_counter()
        print('P-values adjusted: {}s'.format(lr_, end - start))

    adata.obsm['spot_lr_padj'] = pd.DataFrame(
        pvals_adj,
        index=adata.obs_names,
        columns=lr_pairs
    )