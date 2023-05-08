#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@Desc:

"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def detect_svg(adata,
               target_domains=['0', '1', '2'],
               filter_mt=True,
               use_log=False,
               domain_labels='SpaSEG_clusters', do_filter=True,
               max_p=0.05,max_in_p=0.05, min_in_frac=0.8, min_log2fc=1.5, min_log2fc_in_max=1.5, min_in_out_frac=1.,min_in_max_frac=0.95, 
              cut_min_in_out_frac=1.2,cut_min_log2fc=3.0, cut_min_log2fc_in_max=3.0, cut_min_in_max_frac=1.2, cut_max_in_expr_cv=0.6,cut_max_cv_r=1.0):
    
    """


    :param adata:
    :param target_domains: list, or 'all'
        target domains for detecting SVGs
    :param filter_mt: bool,
        filter out MT related genes.
    :param use_log: bool,
        use log1p transformed gene expression.
    :param domain_labels: str,
        domain labels returned by SpaSEG.
    :return:
    """

    adata_ = _data_prep(adata, filter_mt=filter_mt, use_log=use_log)
    
    adata_.obs[domain_labels] = adata_.obs[domain_labels].astype(str).astype('category')
    if target_domains == 'all':
        target_domains = adata_.obs[domain_labels].unique().to_list()
        
    print('rank genes....')
    _rank_gene(adata_, groups=domain_labels)
    print('rank genes done')
    
    re_df = []
    for i, i_domain in enumerate(tqdm(target_domains,desc='Detect SVGs for target domains')):
        t_svgs = _ranks_svg(adata_, domain=i_domain, groups=domain_labels)
        t_svgs['cv_r']=t_svgs['in_expr_cv']/t_svgs['max_expr_cv']
        t_svgs['in_out_fraction'] = t_svgs['in_fraction']/t_svgs['out_fraction'] 
        t_svgs['in_max_fraction'] = t_svgs['in_fraction']/t_svgs['max_fraction'] 
        
        if do_filter:
            re_svgs = _find_svg(t_svgs, max_p=max_p,max_in_p=max_in_p,  min_in_frac=min_in_frac, min_log2fc=min_log2fc, min_log2fc_in_max=min_log2fc_in_max, min_in_out_frac=min_in_out_frac,min_in_max_frac=min_in_max_frac, cut_min_in_out_frac=cut_min_in_out_frac,cut_min_log2fc=cut_min_log2fc, cut_min_log2fc_in_max=cut_min_log2fc_in_max, cut_min_in_max_frac=cut_min_in_max_frac, cut_max_in_expr_cv=cut_max_in_expr_cv,cut_max_cv_r=cut_max_cv_r)
        else:
            re_svgs=t_svgs
        if i == 0:
            re_df = re_svgs
        else:
            re_df = pd.concat([re_df, re_svgs], ignore_index=True)

    return re_df, adata_


def _data_prep(adata, filter_mt=True, use_log=False):
    """
    data preprocess for SVG detection.

    :param adata:
        annoData for spatial transcriptomics.
        gene expression has been normalized.
    :param filter_mt:
        filter out MT related genes
    :param use_log:
        use gene expression after log1p transformation
    :return:
    """

    adata_ = adata.copy()
    adata_.var_names_make_unique()
#     sc.pp.highly_variable_genes(adata_, n_top_genes=3000)
#     adata_ = adata_[:, adata_.var.highly_variable]

    if filter_mt:
        adata_ = adata_[:, ~adata_.var['mt']]

    if not use_log:
        adata_.X = np.expm1(adata_.X.toarray())

    return adata_


def _rank_gene(adata, groups='SpaSEG_clusters'):
    """
    rank gene groups
    :param adata:

    :param groups:
        group labels returned by SpaSEG
    :return:
    """
    adata.obs[groups] = adata.obs[groups].astype(str).astype('category')
    sc.tl.rank_genes_groups(adata,
                            groupby=groups, reference="rest",
                            n_genes=adata.shape[1], method='wilcoxon')


def _ranks_svg(adata, domain='0', groups='SpaSEG_clusters'):
    """
    ranks SVGs for target domain.

    :param adata:
        adata after running sc.tl.rank_genes_groups
    :param domain:
        target domain
    :param groups:
        group label returned by SpaSEG
    :return:
        ranked genes in the target domains, saved in pandas DataFrame
    """
    
    # group='0'
    adata.obs[groups] = adata.obs[groups].astype(str).astype('category')
    out_groups = adata.obs[groups].unique().to_list()
    group = domain
    if isinstance(domain, int):
        group = str(group)
    if isinstance(out_groups[0], int):
        out_groups = [str(x) for x in out_groups]
    out_groups.remove(group)

    genes = adata.uns['rank_genes_groups']['names'][group]
    log2_fc = adata.uns['rank_genes_groups']['logfoldchanges'][group]
    pvals = adata.uns['rank_genes_groups']['pvals_adj'][group]

    adata_df = adata.to_df()
    adata_df = adata_df.astype('float64')

    group_bool = adata_df.copy()[genes]
    adata_df.index = adata.obs[groups]

    df_count_fracs = adata_df.groupby(level=0).apply(lambda x: x.where(x > 0).count() / x.count())[genes].T
    df_count_fracs_out_groups = df_count_fracs.loc[:,out_groups]
    # get max frac in out_groups
    df_count_fracs_out_groups_max = df_count_fracs_out_groups.max(1)
    df_count_fracs_out_groups_max_clusterid = df_count_fracs_out_groups.idxmax(1)

    df_mean_expr = adata_df.groupby(level=0).mean()[genes].T
    df_std_expr = adata_df.groupby(level=0).std()[genes].T
    
    df_mean_expr['max_id'] = df_count_fracs_out_groups_max_clusterid
    df_std_expr['max_id'] = df_count_fracs_out_groups_max_clusterid
    
#     gene_in_max_p = {}
#     for i, i_gene in enumerate(genes):
#         x = adata_df.loc[adata_df.index == group,i_gene].values
#         y = adata_df.loc[adata_df.index == df_count_fracs_out_groups_max_clusterid[i_gene], i_gene].values
#         _, pvalue = ranksums(x, y)
#         _pvals_adj = multipletests(pvalue, method="fdr_bh")[1][0]
#         gene_in_max_p[i_gene] = _pvals_adj
    
    
    df_mean_expr_group = df_mean_expr.loc[:, group].values
    df_mean_expr_group[df_mean_expr_group == 0] = 1e-12
    df_std_expr_group = df_std_expr.loc[:, group].values
    
#     # get max by mean expr
#     df_mean_expr_out_groups2 = df_mean_expr.loc[:,out_groups]
#     mean_expr_out_groups_max2 = df_mean_expr_out_groups2.max(1)
#     mean_expr_out_groups_max_clusterid2 = df_mean_expr_out_groups2.idxmax(1)
#     log2_fc_in_outmax2 = np.log2(
#         (np.expm1(df_mean_expr_group)+1e-9)/(np.expm1(mean_expr_out_groups_max2)+1e-9)
#     )
    
#     df_count_fracs['max_id'] = mean_expr_out_groups_max_clusterid2
#     df_std_expr['max_id'] = mean_expr_out_groups_max_clusterid2
    
#     df_max_frac_out = df_count_fracs.apply(lambda x: x[x['max_id']], axis=1).values
#     df_max_std_expr_out = df_std_expr.apply(lambda x: x[x['max_id']], axis=1).values
    
    df_max_mean_expr_out = df_mean_expr.apply(lambda x: x[x['max_id']], axis=1).values
    df_max_mean_expr_out[df_max_mean_expr_out == 0] = 1e-12
    df_max_std_expr_out = df_std_expr.apply(lambda x: x[x['max_id']], axis=1).values

    log2_fc_in_outmax = np.log2(
        (np.expm1(df_mean_expr_group) + 1e-9) / (np.expm1(df_max_mean_expr_out) + 1e-9)
    )

    group_bool.index = ((adata.obs[groups] == group) * 1).astype('category')
    group_bools = group_bool.astype(bool)
    fraction_obs = group_bools.groupby(level=0).sum() / group_bools.groupby(level=0).count()

    in_fraction = fraction_obs.loc[1]
    out_fraction = fraction_obs.loc[0]
    
    cv_group = df_std_expr_group / df_mean_expr_group
# #     dispersion_group = np.log(dispersion_group)
    
    cv_max = df_max_std_expr_out / df_max_mean_expr_out
# #     dispersion_max = np.log(dispersion_max)
    
# #     disp_r = dispersion_group / dispersion_max

    re_df_group = pd.DataFrame({
        'gene': genes,
        'pvals_adj': pvals,
        'log2_fc': log2_fc,

        'in_fraction': in_fraction.values,
        'out_fraction': out_fraction.values,
#         'in_out_fraction': in_fraction.values/out_fraction.values,
        # 'out_max_fraction': df_count_fracs_out_groups_max.values,
        # 'in_max_fraction': in_fraction.values/df_count_fracs_out_groups_max.values,
        
        'max_fraction': df_count_fracs_out_groups_max,
#         'in_max_fraction': in_fraction.values/df_max_frac_out,

        'in_expr_mean_log': np.log1p(df_mean_expr_group),
        'in_expr_cv':cv_group,
        'max_expr_mean_log': np.log1p(df_max_mean_expr_out),
        
#         'max_out_group_expr_mean_log': np.log1p(df_max_mean_expr_out),
        'max_expr_cv': cv_max,
#         'in_max_dispersion':dispersion_group/dispersion_max,
        
        'max_expr_domain': df_count_fracs_out_groups_max_clusterid.values,
        'log2_fc_in_max': log2_fc_in_outmax
    })

    re_df_group['domain'] = group
#     re_df_group['in_max_p'] = pd.Series(gene_in_max_p)

    return re_df_group


def _find_svg(df_group_r, max_p=0.05,max_in_p=0.05,  min_in_frac=0.75, min_log2fc=1.5, min_log2fc_in_max=1.5, min_in_out_frac=1.,min_in_max_frac=0.95, 
              cut_min_in_out_frac=1.2,cut_min_log2fc=3.0, cut_min_log2fc_in_max=3.0, cut_min_in_max_frac=1.2, cut_max_in_expr_cv=0.6, cut_max_cv_r=1.0, min_high_log2_fc=8):
    """

    :param df_group_r:
    :return:
        SVG sorted by  log2_fc_in_outmax
    """
    df_group = df_group_r.copy()

    
#     df_group=df_group.loc[(df_group['pvals_adj'] < max_p) &
# #                           (df_group['in_max_p'] < max_in_p) &
#                           (df_group['in_fraction'] > min_in_frac) &
#                           (df_group['log2_fc_in_max'] > min_log2fc_in_max) &
#                           (df_group['log2_fc'] > min_log2fc) &
#                           (df_group['in_max_fraction'] > min_in_max_frac) &
#                           (df_group['in_out_fraction'] > min_in_out_frac)
#                          ]
#     cond1_cut = (df_group['in_out_fraction']> cut_min_in_out_frac)
#     cond2_cut = ((df_group['log2_fc']>cut_min_log2fc)&(df_group['log2_fc_in_max']>cut_min_log2fc_in_max)&(df_group['in_max_fraction']>cut_min_in_max_frac)&(df_group['in_expr_cv']<cut_min_in_expr_cv))
    
#     df_group=df_group.loc[cond1_cut|cond2_cut]
    
    pre_req = (
        (df_group['pvals_adj'] < max_p) &
        (df_group['in_fraction'] > min_in_frac) &
        (df_group['log2_fc_in_max'] > min_log2fc_in_max) &
        (df_group['log2_fc'] > min_log2fc) &
        (df_group['in_max_fraction'] > min_in_max_frac) &
        (df_group['in_out_fraction'] > min_in_out_frac)
    )
    df_group=df_group.loc[pre_req]
                     
    cond1_cut = (
        (df_group['in_out_fraction']> cut_min_in_out_frac) | 
        (df_group['in_max_fraction']> cut_min_in_max_frac)
    )
    cond2_cut = (
        (df_group['log2_fc']>cut_min_log2fc) &
        (df_group['log2_fc_in_max']>cut_min_log2fc_in_max) & 
             # (df_group['max_fraction']< 1) & 
        (df_group['in_expr_cv']<cut_max_in_expr_cv)&
        (df_group['cv_r']<cut_max_cv_r)
    )
    
    cond3_cut= (df_group['log2_fc']> min_high_log2_fc) | (df_group['log2_fc_in_max']> min_high_log2_fc) 
    df_group=df_group.loc[cond1_cut|cond2_cut|cond3_cut]
    
#     df_group=df_group.loc[cond1_cut|cond2_cut]

    return df_group.sort_values(by='out_fraction')