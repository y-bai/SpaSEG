#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@File: _heatmapplot.py
@Desc:

"""
import numpy as np
import pandas as pd

import scanpy as sc
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap

from ._utils import _panel_grid, _get_domain_edge, _check_colornorm

centimeter = 1 / 2.54  # one centimeter in inches, 1 inch = 2.54 cm


def gheatmap(adata, genes, domain_key='SpaSEG_clusters', palette=None, fig_w_cm=4.6, fig_h_cm=6, vmin=None, vmax=None, 
              cbar_title='Expr.\n (Z-score)',domain_order=None,yticklabelsize=5):
    """
    heatmap plot with group(or domain) bar
    
    adata: have been scaled (ie, mean=0,std=1) for each genes over all obs.
    genes: {domain_id: gene_list}. 
        eg, {1: [gene1, gene2,..], 2:[gene4, gene5,..]}
    
    """
    domains_s = adata.obs[domain_key]
    if domain_order is None:
        uni_domains = np.sort(domains_s.unique())
    else:
        uni_domains=domain_order

    # get gene list according to the ordered domain
    _genes=[]
    for d in uni_domains:
        if d in genes:
            _genes.extend(genes[d])
    
    _exp_df = adata.to_df().loc[:,_genes]
    _exp_df[domain_key]=domains_s
    
    # re-order dataframe according to the ordered domain
    gene_exp_df = None
    n_obs_domain = []
    for i, _d in enumerate(domain_order):
        _tmp_df = _exp_df[_exp_df[domain_key]==_d]
        n_obs_domain.append(_tmp_df.shape[0])
        
        if i == 0:
            gene_exp_df = _tmp_df
        else: 
            gene_exp_df = pd.concat([gene_exp_df, _tmp_df])
    
    n_domain = len(uni_domains)
    obs_pos = [0] + np.cumsum(n_obs_domain).tolist() #pandas series
    
    _gene_matrix = gene_exp_df.values[:,:-1].T
    
    _vmin = _gene_matrix.min()
    _vmax = _gene_matrix.max()
    if vmin is not None:
        _vmin = vmin
    if vmax is not None:
        _vmax = vmax 
    
    clist=['magenta','black','yellow']
    hmap_cmap = LinearSegmentedColormap.from_list('gene_cmap',clist,N=256)
    
    if palette is None:
        palette = dict(zip(uni_domains,sc.pl.palettes.default_20[:len(uni_domains)]))
    
    fig, axs = plt.subplots(2,n_domain,figsize=(fig_w_cm, fig_h_cm),
                           gridspec_kw={"hspace":0.02,
                                        "wspace":0.03, 
                                        "height_ratios":[1,fig_h_cm*15],
                                        "width_ratios":n_obs_domain}
                           )
    
    for i in range(n_domain):
        # draw top bar
        axs[0,i].set_facecolor(palette[uni_domains[i]])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[0,i].spines['right'].set_visible(False)
        axs[0,i].spines['top'].set_visible(False)
        axs[0,i].spines['left'].set_visible(False)
        axs[0,i].spines['bottom'].set_visible(False)
        axs[0,i].text(0.5,1.8, uni_domains[i],
                   transform=axs[0,i].transAxes,
                   ha="center", va="center")
        
        # draw heatmap
        axs[1,i].imshow(_gene_matrix[:, obs_pos[i]:obs_pos[i+1]], cmap=hmap_cmap, aspect='auto', 
                            norm=_check_colornorm(vmin=_vmin, vmax=_vmax), 
                            interpolation=None) # interpolation='nearest',
        axs[1,i].spines['right'].set_visible(False)
        axs[1,i].spines['top'].set_visible(False)
        axs[1,i].spines['left'].set_visible(False)
        axs[1,i].spines['bottom'].set_visible(False)
        axs[1,i].set_xticks([])
        if i==0:
            axs[1,i].set_yticks(np.array(range(len(_genes))))
            axs[1,i].set_yticklabels(_genes,fontsize=yticklabelsize)
            axs[1,i].tick_params(direction='out', length=1.5, pad=1, width=0.5)
        else:
            axs[1,i].set_yticks([])
    
    # color bar
    cax = fig.add_axes([1.05,0.2,0.03,0.15])
    # fig.colorbar(im, cax=cax)
    fig.colorbar(plt.cm.ScalarMappable(norm=_check_colornorm(vmin=_vmin, vmax=_vmax),cmap=hmap_cmap),
                 cax=cax, orientation='vertical')
    cax.tick_params(direction='out',length=1.5, width=0.5, pad=1,labelsize=5)
    cax.set_frame_on(False)
    cax.set_title(cbar_title, loc='center',fontsize=5, pad=0, y=1.1)