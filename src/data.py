from PIL import Image
import numpy as np
import pandas as pd


# Data processings
from timm.data import create_transform
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode


class PawpularDataset:
    def __init__(
        self,
        img_paths,
        train=True,
        p_new=None,
        targets=None,
        augmentations=None,
        img_paths_new=None,
    ):
        self.img_paths = img_paths
        self.img_paths_new = img_paths_new
        self.augmentations = augmentations

        self.p_new = p_new
        self.train = train
        if train:
            self.targets = targets

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        do_semi_supervided = self.p_new and np.random.rand() < self.p_new
        if do_semi_supervided:
            idx_new = np.random.choice(len(self.img_paths_new))
            image = Image.open(self.img_paths_new[idx_new]).convert("RGB")
            target = -1 * np.ones(1).astype(np.float32)
        elif self.train:
            image = Image.open(self.img_paths[idx]).convert("RGB")
            target = (self.targets[idx]).reshape(-1).astype(np.float32) / 100.0
        else:
            image = Image.open(self.img_paths[idx]).convert("RGB")

        if self.augmentations is not None:
            image = self.augmentations(image)

        if self.train:
            return image, target
        else:
            return (image,)


def get_valid_aug(args):
    """Transforms"""
    # to maintain same ratio w.r.t. 224 images
    size = int(args.input_size / args.crop_pct)
    aug = transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return aug


def get_test_aug(args):
    """Transforms"""
    # to maintain same ratio w.r.t. 224 images
    size = int(args.input_size / args.crop_pct)
    aug = transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return aug


def get_datasets(args):
    """Divides the competition dataset into training and validation sets"""

    df = pd.read_csv(
        args.repo_dir + "/results/csv_files/train_%dfolds.csv" % args.Kfold
    )
    df_train = df[df.kfold != args.fold - 1].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold - 1].reset_index(drop=True)

    train_img_paths = [
        args.data_dir + args.comp_name + f"/train/{x}.jpg"
        for x in df_train["Id"].values
    ]
    valid_img_paths = [
        args.data_dir + args.comp_name + f"/train/{x}.jpg"
        for x in df_valid["Id"].values
    ]

    if args.p_new:
        df_train_new = pd.read_csv(
            args.repo_dir + "/results/csv_files/new_unlabeled_images.csv"
        )
        # print("Number of samples in new dataset: ", len(df_train_new))
        train_img_paths_new = [
            args.data_dir + f"adoption-speeds/{x}" for x in df_train_new["Id"].values
        ]

    valid_aug = get_valid_aug(args)
    valid_dataset = PawpularDataset(
        img_paths=valid_img_paths,
        targets=df_valid.Pawpularity.values,
        augmentations=valid_aug,
    )

    if args.disable_train_aug:
        train_aug = valid_aug
    else:
        train_aug = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=None,
            auto_augment=args.aa,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            crop_pct=args.crop_pct,
            interpolation=InterpolationMode.BICUBIC,
        )

    if args.p_new:
        train_dataset = PawpularDataset(
            img_paths=train_img_paths,
            targets=df_train.Pawpularity.values,
            img_paths_new=train_img_paths_new,
            augmentations=train_aug,
            p_new=args.p_new,
        )
    else:
        train_dataset = PawpularDataset(
            img_paths=train_img_paths,
            targets=df_train.Pawpularity.values,
            augmentations=train_aug,
        )

    return train_dataset, valid_dataset


def get_test_dataset(args):
    df_test = pd.read_csv(args.data_dir + args.comp_name + "/test.csv")
    test_img_paths = [
        args.data_dir + args.comp_name + f"/test/{x}.jpg" for x in df_test["Id"].values
    ]

    test_aug = get_test_aug(args)
    test_dataset = PawpularDataset(
        augmentations=test_aug, img_paths=test_img_paths, train=False
    )
    return test_dataset
