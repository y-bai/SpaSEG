#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@File: _svgplot.py
@Desc:

"""

import scanpy as sc
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from ._utils import _panel_grid, _get_domain_edge

centimeter = 1 / 2.54  # one centimeter in inches, 1 inch = 2.54 cm


def svgplot(anndata,
            genes,
            *,
            axes_w_cm=3.2,
            add_edge=True,
            target_domain=['0'],
            domain_labels='SpaSEG_clusters',
            domain_edge_alpha=100,
            domain_edge_lw=0.5,
            domain_edge_color='k',
            wspace=None,
            hspace=0.25,
            ncols=5,
            img_key=None,
            basis='spatial',
            st_type='10x',
            panel_cbar=True,
            min_max=True,
            cbar_wid_r=0.1,
            cbar_title = 'Expr.',
            fontsize=6,
            **kwargs):
    """
    plot SVGs

    :param anndata:
        ST data with domain labels,
    :param genes:
        SVG genes,
    :param axes_w_cm:
        width of axes, in centimeter,
    :param add_edge:
        whether plot domain edge,
    :param target_domain:
        label of target_domain,
    :param domain_labels:
        refer to the .obs[domain_labels], domains labels are of str/category type,
    :param domain_edge_alpha:
        identify domain edge using alpha shape algorithm,
    :param domain_edge_lw:
        domain edge line width,
    :param domain_edge_color:
        domain edge color
    :param wspace:
        wspace for axes
    :param hspace:
        hspace for axes
    :param ncols:
        the number of axes per row,
    :param img_key:
        img_key in .uns['spatial']
    :param basis:
        coordinates in .obsm['spatial'], np.ndarray, first col: row/height, second col: col/width
    :param st_type:
        spatial transcriptomics plateform,
    :param panel_cbar:
        whether each axes needs color bar,
    :param min_max:
        whether do min max for adata.X
    :param kwargs:
        :kwargs for sc.pl.spatial
    :return:
    """

    adata = anndata.copy()
    
    if min_max:
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        x = adata.X
        if issparse(x):
            x = x.toarray()
        adata.X = min_max_scaler.fit_transform(x)

    _axes_w_inch = axes_w_cm * centimeter  # inch

    if basis not in adata.obsm:
        raise KeyError(f"Could not find '{basis}' in .obsm")

    coords = adata.obsm[basis]  # coords[:,0]:row (or height, y),coords[:,0]:col (or width, x)
    w_h = coords.max(0) - coords.min(0)  # weight and heigh
    # get ratio of width to height of each panel (i.e., axes) 
    w_h_ratio = w_h[0] / w_h[1]
    _axes_h_inch = _axes_w_inch / w_h_ratio

    if not isinstance(genes, list):
        genes = [genes]
    n_panels = len(genes)

    if wspace is None:
        wspace = 0.75 * centimeter / _axes_w_inch + 0.02

    fig, grid = _panel_grid(
        _axes_w_inch, _axes_h_inch, hspace, wspace, ncols, n_panels
    )

    scale_factor = 1.0
    library_id = None
    if img_key is not None and st_type == '10x':
        libraries = adata.uns['spatial'].keys()
        if len(libraries) > 1:
            raise ValueError(
                "Found multiple possible libraries in `.uns['spatial']. Please specify."
            )
        elif len(libraries) == 1:
            library_id = list(libraries)[0]
        else:
            library_id = None

    if library_id is not None and img_key is not None:
        scale_factor = adata.uns['spatial'][library_id]['scalefactors'][f"tissue_{img_key}_scalef"]
        
    axs = []

    for _i in range(n_panels):

        ax = plt.subplot(grid[_i])

        sc.pl.spatial(adata,
                      color=genes[_i], show=False, ax=ax, basis=basis, img_key=img_key, **kwargs)  # ax is exactly axi

        if add_edge:
            domain_coords, domain_edges = _get_domain_edge(adata,
                                                           target_domain=target_domain[_i],
                                                           domain_labels=domain_labels,
                                                           basis=basis, alpha=domain_edge_alpha)

            for i, j in domain_edges:
                ax.plot(domain_coords[[i, j], 0] * scale_factor,
                        domain_coords[[i, j], 1] * scale_factor,
                        linestyle="-",
                        linewidth=domain_edge_lw,
                        color=domain_edge_color)
        if target_domain is None:
            _title = f"$\it {genes[_i]}$"
        else:
            _title = f"$\it {genes[_i]}$: domain {target_domain[_i]}"

        if library_id is None and img_key is None:
            ax.set_title(_title,fontsize=fontsize,pad=0, y=0.98)
        else:
            ax.set_title(_title,fontsize=fontsize)
        ax.figure.get_axes()[-1].remove()  # remove color bar

        if panel_cbar:
            _cbar(ax, min_max, cbar_wid_r=cbar_wid_r, cbar_title=cbar_title)
        else:
            if (_i > 0 and (_i + 1) % ncols == 0) or (_i > 0 and (_i + 1) == n_panels):
                _cbar(ax, min_max, cbar_wid_r=cbar_wid_r, cbar_title = cbar_title)
        
        axs.append(ax)
    
    return axs


def _cbar(ax, min_max, cbar_wid_r=0.1, cbar_title='Expr.', cbar_fontsize=5):
    x0 = ax.get_position().x0
    x1 = ax.get_position().x1
    y0 = ax.get_position().y0
    y1 = ax.get_position().y1

    hei = (y1 - y0) / 4.0

    # cax = ax.figure.add_axes(mtransforms.Bbox.from_extents(
    #     x1 + 0.005,  # fig_left
    #     y0 + hei,    # fig_bottom
    #     x1 + (x1-x0)*cbar_wid_r,  # fig_right
    #     y1 - hei  # fig_top
    # ))
    
    cax = ax.figure.add_axes([
        x1 + 0.005,  # fig_left
        y0 + hei,    # fig_bottom
        (x1-x0)*cbar_wid_r,  # fig_width
        ax.get_position().height * 0.3 # fig_height
    ])

    _cbar = plt.colorbar(ax.collections[0], cax=cax, orientation='vertical', ax=ax)
    _cbar.ax.tick_params(direction='out',length=0., width=0., pad=0.3,labelsize=cbar_fontsize)
    cax.set_frame_on(False)
    cax.set_title(cbar_title, loc='left',fontsize=cbar_fontsize, pad=0, y=1.1)
    if min_max:
        _cbar.set_ticks([
            ax.collections[0].get_array().min()+0.1, 
            ax.collections[0].get_array().max()-0.1
        ])
        _cbar.set_ticklabels(['Min', 'Max'])

# def _spa_plot(adata, genes, basis='spatial', spot_size=1.5, axes_w_cm=3.2, font_size=5, invert_y=True, cmap='coolwarm', add_edge=True, target_domain='0', domain_labels='SpaSEG_clusters', alpha=100, lw=0.5):

#     _axes_w_inch = axes_w_cm * centimeter

#     if basis not in adata.obsm:
#         raise KeyError(f"Could not find '{basis}' in .obsm")

#     coords = adata.obsm['spatial']
#     w_h = coords.max(0) - coords.min(0)  # weight and heigh
#     w_h_ratio = w_h[0] / w_h[1]

#     _axes_h_inch = _axes_w_inch / w_h_ratio

#     wspace = 0.75 / _axes_w_inch + 0.02
#     hspace = 0.25
#     n_cols = 5

#     if not isinstance(genes, list):
#         genes = [genes]
#     num_panels = len(genes)

#     fig, grid = _panel_grid(
#         _axes_w_inch, _axes_h_inch, hspace, wspace, n_cols, num_panels
#     )

#     for _i in range(num_panels):

#         gene_expr = adata[:,genes[_i]].X
#         if issparse(gene_expr):
#             gene_expr = gene_expr.A.squeeze()
#         else:
#             gene_expr = gene_expr.squeeze()

#         ax = plt.subplot(grid[_i])
#         ax.scatter(coords[:, 0], coords[:, 1],
#                    marker='.',
#                    s=spot_size,
#                    c=gene_expr,
#                    cmap=cmap
#                    )

#         if add_edge:
#             domain_coords, domain_edges = _get_domain_edge(adata,
#                      target_domain=target_domain,
#                      domain_labels=domain_labels,
#                      basis=basis, alpha=alpha)

#             for i, j in domain_edges:
#                 ax.plot(domain_coords[[i, j], 0], domain_coords[[i, j], 1],
#                     linestyle="-",
#                     linewidth=lw,
#                     color='black')

#         ax.axis('off')
#         ax.set_aspect("equal")
#         if invert_y:
#             ax.invert_yaxis()
