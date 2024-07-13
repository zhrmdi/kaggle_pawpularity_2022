import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.data import get_datasets
from torch.utils.data import DataLoader
from torchmetrics import Metric
from src.scheduler import CustomLRScheduler


class MyRMSE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.squared_error += torch.sum((target - preds) ** 2)
        self.total += target.numel()

    def compute(self):
        return 100 * torch.sqrt(self.squared_error / self.total)


class PawpularModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.testing = args.testing
        self.automatic_optimization = True

        self.train_dataset, self.valid_dataset = get_datasets(self.args)

        self.encoder = timm.create_model(
            args.model,
            in_chans=3,
            num_classes=0,
            drop_path_rate=0,
            drop_rate=args.drop_path,
            pretrained=not self.testing,
        )

        num_features = self.encoder.num_features

        self.dropout = nn.Dropout(args.dropout)

        # mean estimator head
        self.fc_mean = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid(),
        )

        # var estimator head
        self.fc_var = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid(),
        )

        self.train_rmse = MyRMSE()
        self.valid_rmse = MyRMSE()
        self.test_rmse = MyRMSE()

    def forward(self, batch, compute_loss=False):
        imgs = batch[0]

        x = self.dropout(self.encoder(imgs))
        vars = self.fc_var(x)
        preds = self.fc_mean(x)

        if compute_loss:
            targets = batch[1]
            loss = self.compute_loss(preds, vars, targets)
            return preds, targets, loss
        return preds, vars

    def compute_loss(self, preds, vars, targets, min_var=1e-3):
        semi_supervised_idxs = targets == -1.0
        targets[semi_supervised_idxs] = preds[semi_supervised_idxs] + 0.175

        errors = targets - preds
        errors_loss = torch.log(torch.cosh(errors))
        # preventing vars to go bellow a min value
        # in order to avoid numerical problems in log
        epsilon = torch.ones_like(preds) * min_var
        vars = torch.maximum(epsilon, vars)
        loss = errors_loss / vars + torch.log(vars)
        loss = torch.mean(loss)
        return loss

    def configure_optimizers(self):
        import torch_optimizer as optim

        optimizer = eval(self.args.opt)(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        scheduler = CustomLRScheduler(optimizer, self.args)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        preds, targets, loss = self.forward(batch, compute_loss=True)
        self.train_rmse(preds, targets)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch == self.args.fc_epochs - 1:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("Encoder is unleashed!")

        self.log("learning_rate", self.lr_schedulers().get_last_lr()[-1])

    def validation_step(self, batch, batch_idx):
        preds, targets, loss = self.forward(batch, compute_loss=True)
        self.valid_rmse(preds, targets)
        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True)
        self.log("valid_rmse", self.valid_rmse, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        preds, targets, loss = self.forward(batch, compute_loss=True)
        self.test_rmse(preds, targets)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, self.args.batch_size, self.args)

    def get_dataloader(self, train_dataset, batch_size, args):
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()


# Test
if __name__ == "__main__":
    from src.pawpular_training import get_args

    args = get_args()
    model = PawpularModel(args)
    model.configure_optimizers()
    batch = torch.randn(1, 3, 224, 224), torch.randn(1, 1, 224, 224)
    out = model(batch)
    print(out)
