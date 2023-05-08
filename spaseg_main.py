#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import time
import torch
import scanpy as sc

from spaseg import spaseg
from data_processing.scanpy_processing import sc_processing

def main(in_args):
    experiment_name = in_args.experiment_name
    st_platform = in_args.platform
    use_gpu = in_args.use_gpu
    gpu_device = in_args.gpu_device
    in_file_path = in_args.in_file_path
    sample_id_list = in_args.sample_id_list
    preprocess = in_args.scanpy_process
    drop_cell_ratio = in_args.drop_cell_ratio
    min_cells = in_args.min_cells
    compons = in_args.compons
    cache_file_path = in_args.cache_file_path
    out_file_path = in_args.out_file_path
    input_dim = in_args.input_dim
    nChannel = in_args.nChannel
    output_dim = in_args.output_dim
    lr = in_args.lr
    nConv = in_args.nConv
    pretrain_epochs = in_args.pretrain_epochs
    alpha = in_args.alpha
    beta = in_args.beta
    min_label = in_args.minLabel
    ground_truth_index = in_args.ground_truth_index
    spot_size = in_args.spot_size

    if experiment_name:
        out_file_path = os.path.join(out_file_path, experiment_name)
        if not os.path.isdir(out_file_path):
            os.mkdir(out_file_path)

    sc.settings.figdir = out_file_path

    n_batch = len(sample_id_list)
    multi_slice = n_batch>1

    # preprocessing data
    if preprocess:
        in_file_path_list = [os.path.join(in_file_path, "{}.h5ad".format(sample_id)) for sample_id in
                             sample_id_list]

        adata_list = [sc.read_h5ad(file_path) for file_path in in_file_path_list]
        adata_list = sc_processing(adata_list,
                      sample_id_list,
                      multi_slice,
                      st_platform,
                      drop_cell_ratio,
                      min_cells,
                      compons)

    else:
        file_path_list = [os.path.join(cache_file_path, "preprocessed_{}.h5ad".format(sample_id)) for sample_id in
                      sample_id_list]

        adata_list = [sc.read_h5ad(file_path) for file_path in file_path_list]

    start = time.time()
    torch.cuda.empty_cache()
    spaseg_model = spaseg.SpaSEG(adata=adata_list,
                                 use_gpu=use_gpu,
                                 device=gpu_device,
                                 input_dim=input_dim,
                                 nChannel=nChannel,
                                 output_dim=output_dim,
                                 nConv=nConv,
                                 lr=lr,
                                 pretrain_epochs=pretrain_epochs,
                                 sim_weight=alpha,
                                 con_weight=beta,
                                 min_label=min_label,
                                 spot_size=spot_size)
    input_mxt, H, W = spaseg_model._prepare_data()
    cluster_label, embedding = spaseg_model._train(input_mxt)
    if multi_slice:
        spaseg_model._add_embedding(H, W, embedding, n_batch)
    spaseg_model._add_seg_label(cluster_label, n_batch, H, W, in_args.barcode_index)
    spaseg_model._cal_metrics(ground_truth_index)
    spaseg_model._save_result(out_file_path, sample_id_list, multi_slice)

    end = time.time()
    print('running time = {}'.format(end - start))


def parse_opt():
    parser = argparse.ArgumentParser(description='SpaSEG model')
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--platform", type=str, default="Visium")
    parser.add_argument("--Stereo_data_type", type=str, default=None,
                        help="Specify Stereo-seq input data types: cellBin or binCell")
    parser.add_argument("--in_file_path", type=str, default="./data/HumanPilot", required=False)
    parser.add_argument("--cache_file_path", type=str, default="./cache", required=False)
    parser.add_argument("--sample_id_list", type=str, nargs='+', required=False, help="Enter a list of input sample")
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch_size')
    parser.add_argument("--min_cells", type=int, default=5)
    parser.add_argument("--compons", type=int, default=15)
    parser.add_argument("--drop_cell_ratio", type=float, default=0.05, required=False)

    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--out_file_path", type=str, default="./result", required=False)
    parser.add_argument("--processed_file_path", type=str, default="./cache/preprocessed_151673.h5ad", required=False)

    parser.add_argument("--gpu_device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pretrain_epochs", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=2100)
    # in single
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.7)

    # in multi
    # parser.add_argument("--sim_weight", type=float, default=0.2)
    # parser.add_argument("--con_weight", type=float, default=0.4)

    parser.add_argument("--minLabel", type=int, default=None)

    parser.add_argument("--input_dim", type=int, default=15)
    parser.add_argument("--nChannel", type=int, default=15)
    parser.add_argument("--nConv", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=15)

    parser.add_argument("--ground_truth_index", type=str, default=None,
                        help="specify the true label index of the spots")
    parser.add_argument("--barcode_index", type=str, default="index", help="specify the barcode index of the spots")
    parser.add_argument("--spot_size", type=float, default=None)
    parser.add_argument("--scanpy_process", action="store_true", default=False, required=False)
    parser.add_argument("--use_gpu", action="store_true", default=False, required=False)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    # parse arg
    opt = parse_opt()
    main(opt)
