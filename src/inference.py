from src.model import PawpularModel
from src.data import get_test_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import time
import os
import gc


def get_preds(model, test_loader):
    preds = []
    preds_var = []
    trainer = pl.Trainer(gpus=1, deterministic=False)
    model_preds = trainer.predict(model, test_loader)
    for batch_preds in model_preds:
        means, vars = batch_preds
        means = means * 100
        preds += means.detach().cpu().numpy().flatten().tolist()
        preds_var += vars.detach().cpu().numpy().flatten().tolist()

    preds = np.array(preds)
    preds_var = np.array(preds_var)
    return preds, preds_var


def get_avg_preds(preds, vars, weighted=False):
    vars = np.array(vars)
    vars_average = np.mean(vars, axis=0)
    preds = np.array(preds)
    if weighted:
        stds = np.sqrt(vars)
        certainties = preds / stds
        sum_certainties = np.sum(certainties, axis=0, keepdims=True)
        weights = certainties / sum_certainties
        preds_average = np.sum(preds * weights, axis=0)
    else:
        preds_average = np.mean(preds, axis=0)
    return preds_average, vars_average


def get_test_dataloader(test_dataset):
    test_dataset = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )
    return test_dataset


def get_one_model_preds(
    args,
    ensemble,
    ckp_path,
    super_final_preds,
    super_final_preds_var,
):
    args.model = ensemble["model"]
    args.input_size = ensemble["input_size"]
    test_datadoader = get_test_dataloader(get_test_dataset(args))
    model = PawpularModel.load_from_checkpoint(args=args, checkpoint_path=ckp_path)

    final_preds, final_preds_var = get_preds(model, test_datadoader)
    super_final_preds += [final_preds]
    super_final_preds_var += [final_preds_var]


def predict(args):
    t = 0
    avg_time = 0
    super_final_preds = []
    super_final_preds_var = []
    for _ in range(args.instance_repeats):
        tic = time.time()
        for ensemble in args.ensembles:
            for checkpoint in os.listdir(ensemble["ckpt_paths"]):
                if "ckpt" in checkpoint:
                    ckp_path = ensemble["ckpt_paths"] + checkpoint
                    print(checkpoint, "is loaded")
                    get_one_model_preds(
                        args,
                        ensemble,
                        ckp_path,
                        super_final_preds,
                        super_final_preds_var,
                    )
                    gc.collect()
        toc = time.time()
        prev_t = t
        t = toc - tic
        avg_time += (t - prev_t) / args.instance_repeats
        print("Average time: ", int(avg_time))

    avg_preds, avg_vars = get_avg_preds(
        super_final_preds, super_final_preds_var, weighted=args.weighted
    )

    df_test = pd.read_csv(args.data_dir + args.comp_name + "/test.csv")
    df_test["Pawpularity"] = avg_preds
    # df_test["Variance"] = avg_vars
    # df_test = df_test[["Id", "Pawpularity", "Variance"]]
    df_test = df_test[["Id", "Pawpularity"]]
    df_test.to_csv("submission.csv", index=False)
    df_test.tail()


# Test
if __name__ == "__main__":
    from src.pawpular_training import get_args

    args = get_args()
    args.testing = True
    args.instance_repeats = 3
    args.ensembles = [
        {
            "model": "swin_large_patch4_window7_224",
            "ckpt_paths": args.data_dir + "/kradz-checkpoints/",
            "input_size": 224,
        },
        {
            "model": "swin_large_patch4_window12_384",
            "ckpt_paths": args.data_dir + "/kradz-checkpoints-two/",
            "input_size": 384,
        },
    ]

    predict(args)
