#!/usr/bin/env python
# coding:UTF-8
import os
import argparse
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from scanpy import logging as logg
import anndata as ad
from anndata import AnnData

from typing import Optional, Sequence
from data_processing._constant import PositionScale, CellbinSize, SpotSize

# sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="Visium")
    parser.add_argument("--Stereo_data_type", type=str, default=None, help="Specify Stereo-seq input data types: cellBin or binCell")
    parser.add_argument("--in_file_path", type=str, default="./data/HumanPilot", required=False)
    parser.add_argument("--cache_file_path", type=str, default="./cache", required=False)
    parser.add_argument("--sample_id", type=str, default="151673", required=False)
    parser.add_argument("--min_cells", type=int, default=5)
    parser.add_argument("--compons", type=int, default=15)
    parser.add_argument("--cellBin_size", type=int, default=14, required=False)
    parser.add_argument("--drop_cell_ratio", type=float, default=0.05, required=False)

    opt = parser.parse_args()
    return opt

def sc_processing(adata_list: Sequence[AnnData],
                  sample_id_list: Sequence[str],
                  multi_slice: bool,
                  st_platform: str,
                  drop_cell_ratio: float = 0.05,
                  min_cells: int = 5,
                  compons: int = 15,
                  save: bool = True,
                  cache_file_path: Optional[str] = "./cache"):
    """
    function description:
        perform data pre-processing for the raw h5ad data from 10X genomics, Stereo-seq, seqFISH, MERFISH, and Slide-SeqV2
        and prepare the image-like tensor input for SpaSEG.
    ----------
    Parameters:
        adata_list: input adata for SpaSEG, either single adata object or multiple adatas.
        sample_id_list: sample identifier for each adata.
        multi_slice: check whether the inputs are multiple slices.
        st_platform: the spatial transcriptomics platform.
        drop_cell_ratio: the fraction that how many bin/spots should be eliminated in our data.
        min_cells: remove low expressed genes, which will be retained if at least expressed in specified number of bin/spots.
        compons: the number of principle components in dimensional reduction.
        save: whether to save pre-processed h5ad file.
        cache_file_path: the file path for saving files.
    ----------
    Returns
        there is no return in this function, we save the adata object into the cache fold and later will reload the
        adata if needed.
    """
    # input_file = os.path.join(opt.in_file_path, "{}.h5ad".format(opt.sample_id))
    logg.info("----------- Start processing input adata: ------------ \n")

    logg.info("Filtering adata according to the min_gene and min_cell threshold \n")
    adata_list = [filter_data(adata, st_platform, drop_cell_ratio, min_cells, sample_id) for sample_id, adata in zip(sample_id_list,adata_list)]
    adata_list = [add_spot_pos(adata, st_platform) if st_platform not in "Visium" else adata for adata in adata_list]

    if multi_slice:
        logg.info("Start multi-slice adata integration and processing")
        adata_map = {sample_id: adata for sample_id, adata in zip(sample_id_list, adata_list)}
        adata = ad.concat(adata_map, join="inner", index_unique="_", label="batch")
    else:
        logg.info("\n----------- Start single-slice adata data processing -----------\n")
        print(adata_list[0])
        adata = adata_list[0]
        adata.obs["batch"] = sample_id_list[0]

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=compons, random_state=0)

    # unpacked the adatas in to single adata list
    pre_adata_list = []
    for sample_id in sample_id_list:
        pre_adata_list.append(adata[adata.obs["batch"] == sample_id])

    if save:
        for adata, sample_id in zip(pre_adata_list, sample_id_list):
            adata.write_h5ad("{}/preprocessed_{}.h5ad".format(cache_file_path, sample_id))

    return pre_adata_list




def filter_data(adata, platform, drop_cell_ratio, min_cells, sample):
    # Reference link:
    # https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
    # QC and counts data normalization

    logg.info(f'\nStart to filter the data for sample {sample}')
    if (platform in ["Visium", "Stereo-seq", "Slide-seqV2"]):
        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        logg.info("the number of spots after filtering min_genes: {}.".format(adata.obs.shape[0]))
    else:
        pass

    if platform in ["Stereo-seq", "Slide-seqV2"]:
        n_min_genes = adata.obs["n_genes_by_counts"].quantile(q=drop_cell_ratio)
        sc.pp.filter_cells(adata, min_genes=n_min_genes)

    sc.pp.filter_genes(adata, min_cells=min_cells)

    return adata

def add_spot_pos(adata, platform):
    adata.obs["array_row"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_col"] = adata.obsm["spatial"][:, 1]

    if np.min(adata.obsm["spatial"]) < 0:
        adata.obs['array_col'] = (adata.obs['array_col'].values - adata.obs['array_col'].values.min())
        adata.obs['array_row'] = (adata.obs['array_row'].values - adata.obs['array_row'].values.min())

    max_coor = np.max(adata.obsm["spatial"])

    """
    The scale factor refer to the code in stLearn:
    https://github.com/BiomedicalMachineLearning/stLearn/blob/master/stlearn/wrapper/read.py
    """

    if platform in ["Stereo-cellBin"]:
        scale = 1.0 / CellbinSize.CELLBIN_SIZE
        adata.uns["spot_size"] = SpotSize.STEREO_SPOT_SIZE
        # adata.obs['array_col'] = adata.obs['array_col'] / opt.cellBin_size
        # adata.obs['array_col'] = adata.obs['array_row'] / opt.cellBin_size
    elif platform == "MERFISH":
        scale = PositionScale.MERFISH_SCALE_FACTOR / max_coor
        adata.uns["spot_size"] = SpotSize.MERFISH_SPOT_SIZE

    elif platform == "seqFISH":
        scale = PositionScale.SEQFISH_SCALE_FACTOR / max_coor
        adata.uns["spot_size"] = SpotSize.SEQFISH_SPOT_SIZE

    elif platform == "Slide-seqV2":
        scale = PositionScale.SLIDESEQV2_SCALE_FACTOR / max_coor
        adata.uns["spot_size"] = SpotSize.SLIDESEQV2_SPOT_SIZE

    elif platform == "Stereo-seq":
        scale = 1
        adata.uns["spot_size"] = SpotSize.STEREO_SPOT_SIZE

    else:
        scale = 1
        adata.uns["spot_size"] = SpotSize.VISIUM_SPOT_SIZE

    adata.obs['array_col'] = adata.obs['array_col'] * scale
    adata.obs['array_row'] = adata.obs['array_row'] * scale
    adata.obsm['spatial'][:, 0] = adata.obs['array_row'].values
    adata.obsm['spatial'][:, 1] = adata.obs['array_col'].values
    
    # logger.info("adata.obsm['spatial']:", adata.obsm["spatial"])
    # logger.info("adata.obs['array_row','array_col']:",adata.obs[['array_row', "array_col"]])

    return adata


def counts_plot(adata):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
    sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
    sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])


def umap_plot(adata):
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)


def spatial_plot(adata):
    sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5)

if __name__ == '__main__':
    # parse arg
    opt = parse_opt()
    sc_processing(opt)