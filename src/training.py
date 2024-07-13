import copy
import time
import wandb
import pytorch_lightning as pl
from src.model import PawpularModel
from pytorch_lightning.plugins import DDPPlugin


def train_test(args):
    timestamp = 1000000 * time.time()
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = "%d - %s" % (timestamp, args.model)

    if args.finetunning:
        model = PawpularModel(args)
        # experiment tracker (you need to sign in with your account)
        wandb_logger = pl.loggers.WandbLogger(
            run_id="%s-%dfold-%d" % (args.model, args.fold, timestamp),
            name="%s <- fold%d" % (args.exp_name, args.fold),
            group=exp_name,
            log_model=True,  # save best model using checkpoint callback
            project=args.project_name,
            entity="arashash_fall2021",
            save_dir=args.save_dir,
            config=args,
        )
        callbacks = []
        checkpoint = pl.callbacks.ModelCheckpoint(
            filename="best_valid_rmse",
            monitor="valid_rmse",
            save_top_k=1,
            mode="min",
            save_last=False,
            save_weights_only=True,
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        )
        callbacks += [checkpoint]

        early_stop = pl.callbacks.EarlyStopping(
            monitor="valid_rmse",
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min",
        )
        callbacks += [early_stop]

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=args.fc_epochs + args.warmup_epochs + args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            strategy=DDPPlugin(find_unused_parameters=False),
        )

        # Train
        trainer.fit(model)

        # Test
        print("Best model path: ", checkpoint.best_model_path)
        model = PawpularModel.load_from_checkpoint(
            args=args, checkpoint_path=checkpoint.best_model_path
        )
        trainer.test(model)
        wandb.finish()

    if args.stochastic_weight_avg:
        timestamp = 1000000 * time.time()
        swa_args = copy.deepcopy(args)
        # unleash the encoder ASAP
        swa_args.fc_epochs = 1
        swa_args.lr = swa_args.lr_min
        swa_args.init_fc_lr = swa_args.lr_min
        swa_args.disable_train_aug = True
        swa_args.opt = "torch.optim.SGD"
        swa_args.project_name = args.project_name + " with SWA"
        run_id = "%s-%dfold-%d" % (args.model, args.fold, timestamp)
        wandb_logger = pl.loggers.WandbLogger(
            id=run_id,
            name="%s <- fold%d" % (swa_args.exp_name, swa_args.fold),
            group=exp_name,
            log_model=True,  # save best model using checkpoint callback
            project=swa_args.project_name,
            entity="arashash_fall2021",
            save_dir=swa_args.save_dir,
            config=swa_args,
        )
        swa = pl.callbacks.StochasticWeightAveraging(
            annealing_epochs=swa_args.swa_warmup_epochs,
            swa_lrs=swa_args.lr_swa,
            swa_epoch_start=1,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=swa_args.devices,
            logger=wandb_logger,
            callbacks=[swa],
            max_epochs=args.swa_warmup_epochs + swa_args.swa_epochs,
            gradient_clip_val=swa_args.gradient_clip_val,
            accumulate_grad_batches=swa_args.accumulate_grad_batches,
            strategy=DDPPlugin(find_unused_parameters=False),
        )
        if args.finetunning:
            model = PawpularModel.load_from_checkpoint(
                args=swa_args, checkpoint_path=checkpoint.best_model_path
            )
        else:
            model = PawpularModel.load_from_checkpoint(
                args=swa_args, checkpoint_path=args.checkpoint
            )
        trainer.fit(model)

        trainer.test(model)

        trainer.save_checkpoint(args.save_dir + "%s.ckpt" % run_id)
        wandb.finish()
