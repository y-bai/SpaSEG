#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@File: _utils.py
@Desc:

"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def _panel_grid(fig_w, fig_h, hspace, wspace, ncols, num_panels):
    """
    creat figure grids

    reference: scanpy

    :param fig_w: width of subplot
    :param fig_h:
    :param hspace:
    :param wspace:
    :param ncols: the number of subplot on a row
    :param num_panels: total number of subplots
    :return:
    """
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = plt.figure(
        figsize=(
            n_panels_x * fig_w * (1 + wspace),
            n_panels_y * fig_h,
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _get_domain_edge(adata,
                     target_domain='0',
                     domain_labels='SpaSEG_clusters',
                     basis='spatial', alpha=100):

    _target_adata = _domain_adata(
        adata, target_domain=target_domain, domain_labels=domain_labels
    )
    if basis not in _target_adata.obsm:
        raise KeyError(f"Could not find '{basis}' in .obsm")

    coords = _target_adata.obsm['spatial']
    edges = _alpha_shape(coords, alpha=alpha, only_outer=True)

    return coords, edges


def _domain_adata(adata, target_domain='0', domain_labels='SpaSEG_clusters'):
    """

    :param adata:
    :param target_domain:
    :param domain_labels:
    :return:
    """

    adata.obs[domain_labels] = adata.obs[domain_labels].astype(str).astype('category')
    index_target = np.where(adata.obs[domain_labels] == target_domain)[0]
    barcode_target = adata.obs[domain_labels].index[index_target]
    target_adata = adata[barcode_target, :]

    return target_adata


def _alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return

        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle 三角形角点的索引
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def _check_colornorm(vmin=None, vmax=None, vcenter=None, norm=None):
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        # matplotlib<3.2
        from matplotlib.colors import DivergingNorm as DivNorm

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            raise ValueError('Passing both norm and vmin/vmax/vcenter is not allowed.')
    else:
        if vcenter is not None:
            norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    return norm
