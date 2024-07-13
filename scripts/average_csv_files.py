csv_files_dir = "./raw_csv_files/"
import os
import re
import pandas as pd

dfs = []
Kfold = 5
dfs_counts = [0] * Kfold

df_ids = pd.read_csv("./results/csv_files/new_unlabeled_images.csv")
for k in range(Kfold):
    df = pd.DataFrame()
    df["id"] = df_ids["Id"].values
    df["avg_pawpularity"] = 0
    df["avg_var"] = 0
    dfs.append(df)

for csv_file in os.listdir(csv_files_dir):
    numbers = re.findall(r"\d+", csv_file)
    if len(numbers) == 2:
        fold, iter = int(numbers[0]), int(numbers[1])
        df_raw = pd.read_csv(csv_files_dir + csv_file)
        df = dfs[fold - 1]
        df["avg_pawpularity"] += df_raw["avg_pawpularity"]
        df["avg_var"] += df_raw["avg_var"]
        dfs_counts[fold - 1] += 1

for k in range(Kfold):
    df = dfs[k]
    count = dfs_counts[k]
    df["avg_pawpularity"] /= count
    df["avg_var"] /= count
    df.to_csv("./results/csv_files/adoption_preds_fold%d.csv" % (k + 1), index=False)
