import timm
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from psutil import cpu_count
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


parser = argparse.ArgumentParser(description="Pseudo-labeling new dataset")
parser.add_argument("--data_dir", default="/data/users/arash/datasets", type=str)
parser.add_argument("--repo_dir", default="..", type=str)
# model
parser.add_argument("--model", default="swin_large_patch4_window7_224", type=str)
parser.add_argument("--dropout", default=0.5, type=float)
# batch loader
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--prefetch_factor", default=32, type=int)
# data transforms
parser.add_argument("--crop_pct", default=0.9, type=float)
parser.add_argument("--interpolation", default="bicubic", type=str)
parser.add_argument("--image_size", default=224, type=int)
# data selection
parser.add_argument("--Kfold", default=5, type=int)
parser.add_argument("--fold", default=1, type=int)
parser.add_argument("--iter", default=1, type=int)
# Hardware
parser.add_argument("--num_workers", default=16, type=int)
parser.add_argument("--device", default=0, type=int)
cfg = parser.parse_args()

checkpoint_path = cfg.repo_dir + "/checkpoints/"
# checkpoints in fold 1 to 5 order
checkpoints = [
    "77981fec4670458821e7f62570a88b72",
    "700ba472372fa1966fda2413d00777e9",
    "90c47ba35c15956a5ac3072d2533f778",
    "500bff1508a9ebff59d9b4ea4252044e",
    "9bde0321d9d8067302663b504ed7424b",
]

test_aug = transforms_imagenet_eval(
    img_size=cfg.image_size,
    crop_pct=cfg.crop_pct,
    interpolation=cfg.interpolation,
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
)
test_aug.transforms[1] = transforms.Compose(
    [
        transforms.RandomCrop(size=(cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)


class TestAdoptionDataset:
    def __init__(self, test_img_ids, augmentations):
        self.augmentations = augmentations
        self.test_img_ids = test_img_ids
        self.path = cfg.data_dir + "/adoption-speeds/"

    def __len__(self):
        return len(self.test_img_ids)

    def __getitem__(self, idx):
        image = Image.open(self.path + self.test_img_ids[idx]).convert("RGB")

        if self.augmentations is not None:
            image = self.augmentations(image)

        return (image,)


def get_test_dataloader(test_dataset):
    test_dataset = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True,
    )
    return test_dataset


def get_avg_preds(preds, vars):
    vars = torch.tensor(np.array(vars))
    orig_preds = torch.tensor(np.array(preds))
    peak_average = torch.mean(orig_preds, dim=0)
    vars_average = torch.mean(vars, dim=0)
    return peak_average.numpy(), vars_average.numpy()


def get_preds(model, test_loader):
    preds = []
    preds_var = []

    trainer = pl.Trainer(devices=[cfg.device], accelerator="gpu")
    model_preds = trainer.predict(model, test_loader)
    for batch_preds in model_preds:
        means, vars = batch_preds
        means = means * 100
        preds += means.detach().cpu().numpy().flatten().tolist()
        preds_var += vars.detach().cpu().numpy().flatten().tolist()

    preds = np.array(preds)
    preds_var = np.array(preds_var)
    return preds, preds_var


class TestPawpularModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg

        self.backbone = timm.create_model(
            self.cfg.model,
            drop_path_rate=0,
            drop_rate=0,
            pretrained=False,
            num_classes=0,
            in_chans=3,
        )
        num_features = self.backbone.num_features
        self.dropout = nn.Dropout(cfg.dropout)

        # mean estimator head
        self.fc_mean = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid(),
        )

        # var estimator head
        self.fc_var = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

        self.loss_fn = nn.GaussianNLLLoss()

        self.train_losses = []
        self.train_scores = []

        self.valid_losses = []
        self.valid_scores = []

    def forward(self, batch):
        imgs = batch[0]
        x = self.dropout(self.backbone(imgs))
        preds = self.fc_mean(x)
        vars = self.fc_var(x)
        return preds, vars


df_test = pd.read_csv(cfg.repo_dir + "/results/csv_files/new_unlabeled_images.csv")
test_img_ids = df_test["Id"].values
test_dataset = TestAdoptionDataset(augmentations=test_aug, test_img_ids=test_img_ids)
test_loader = get_test_dataloader(test_dataset)
checkpoint = checkpoints[cfg.fold - 1]


for i in range(100):
    iter = i + 1
    # iter = cfg.iter
    super_final_preds = []
    super_final_preds_var = []
    model = TestPawpularModel.load_from_checkpoint(
        cfg=cfg, checkpoint_path=checkpoint_path + checkpoint
    )
    final_preds, final_preds_var = get_preds(model, test_loader)
    super_final_preds += [final_preds]
    super_final_preds_var += [final_preds_var]

    avg_preds, avg_vars = get_avg_preds(super_final_preds, super_final_preds_var)

    df_test = pd.DataFrame()
    df_test["id"] = test_img_ids
    df_test["avg_pawpularity"] = avg_preds
    df_test["avg_var"] = avg_vars
    df_test.to_csv(
        cfg.repo_dir
        + "/raw_csv_files/adoption_preds_fold%d_iter%d.csv" % (cfg.fold, iter),
        index=False,
    )
    df_test.tail()
