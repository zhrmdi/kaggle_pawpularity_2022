# -*- coding: utf-8 -*-
"""Training script for the Pawpularity competition in Kaggle: 
https://www.kaggle.com/c/petfinder-pawpularity-score/overview/description
"""

import src
import argparse
import numpy as np

import os
import torch.distributed as dist


def get_args():
    data_dir = "/data/users/arash/datasets/"
    repo_dir = "/data/users/arash/projects/Pawpularity/"
    parser = argparse.ArgumentParser(description="Training pawpular models")
    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--comp_name", default="petfinder-pawpularity-score", type=str)
    parser.add_argument("--project_name", default="pawpularity", type=str)
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--repo_dir", default=repo_dir, type=str)
    parser.add_argument("--save_dir", default=data_dir + "tmp/", type=str)
    parser.add_argument("--slurm_array_id", default=0, type=int)
    # model name
    # parser.add_argument("--model", default="swin_large_patch4_window12_384", type=str)
    parser.add_argument("--model", default="swin_large_patch4_window7_224", type=str)
    # parser.add_argument("--model", default="swin_large_patch4_window12_384_in22k", type=str)
    # parser.add_argument("--model", default="swin_large_patch4_window7_224_in22k", type=str)

    # input size
    parser.add_argument("--input_size", default=224, type=int)
    # parser.add_argument("--input_size", default=384, type=int)

    # batch size
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--batch_size", default=16, type=int)

    # extra model params
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--testing", default=False, type=bool)
    # scheduler
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--init_fc_lr", default=1e-3, type=float)
    parser.add_argument("--lr_min", default=1e-6, type=float)
    parser.add_argument("--lr_swa", default=1e-5, type=float)
    parser.add_argument("--fc_epochs", default=20, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--swa_epochs", default=40, type=int)
    parser.add_argument("--swa_warmup_epochs", default=30, type=int)
    parser.add_argument("--T_0", default=20, type=int)
    # optmiztion
    parser.add_argument("--opt", default="optim.AdamP", type=str)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--finetunning", default=True, type=bool)
    parser.add_argument("--stochastic_weight_avg", default=False, type=bool)
    # batch loader
    parser.add_argument("--prefetch_factor", default=8, type=int)
    # data transforms
    parser.add_argument("--crop_pct", default=0.875, type=float)
    parser.add_argument("--disable_train_aug", default=False, type=bool)
    # Regularizations
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--gradient_clip_val", default=None, type=float)
    # * Augmentations
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    # data selection
    parser.add_argument("--p_new", default=0.2, type=float)  # <-------- to be optimized
    parser.add_argument("--Kfold", default=5, type=int)
    parser.add_argument("--fold", default=1, type=int)
    # Hardware
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--devices", default=[2, 3], type=int, nargs="+")
    # Test time params
    parser.add_argument("--instance_repeats", default=10, type=int)
    parser.add_argument("--weighted", default=False, type=bool)
    parser.add_argument("--ensembles", default=[], type=list)

    args, unknown = parser.parse_known_args()
    return args


def main():
    args = get_args()

    # hyper_name = "fold"
    # hyper_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    hyper_name = "p_new"
    hyper_list = np.linspace(start=0.0, stop=0.9, num=10)

    # args.checkpoint = "./checkpoints/best_valid_rmse.ckpt"
    effective_bs = args.batch_size * args.accumulate_grad_batches * len(args.devices)
    print("Training with effective batchsize: ", effective_bs)
    args.exp_name = "lr=1e-5, init_lr=1e-3, %s" % args.model
    args.project_name = "p_new 5CV search, batchsize=%d" % effective_bs
    if args.slurm_array_id:
        hyper_value = hyper_list[args.slurm_array_id - 1]
        exec(f"args.{hyper_name} = {hyper_value}")
        args.exp_name += f" {hyper_name}={hyper_value}"
        src.training.train_test(args)
    else:
        for hyper_value in hyper_list:
            exec(f"args.{hyper_name} = {hyper_value}")
            args.exp_name += f" {hyper_name}={hyper_value}"

            try:
                src.training.train_test(args)
            except KeyboardInterrupt:
                print("Interrupted")
                try:
                    dist.destroy_process_group()
                except KeyboardInterrupt:
                    os.system(
                        "kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') "
                    )
                break


if __name__ == "__main__":
    main()
