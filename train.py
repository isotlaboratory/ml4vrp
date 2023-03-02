"""# -*- coding: utf-8 -*-."""
import argparse
import glob
import os
import signal

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.trainer import Trainer

import modules


def main(config):
    args = config
    if args.seed is not None:
        seed_everything(args.seed)

    callbacks = [
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.3f}",
            monitor="valid_loss",
            mode="min",
            dirpath=args.log_dir,
            save_top_k=1,
        ),
        ModelCheckpoint(
            filename="latest-{epoch}-{step}",
            monitor="step",
            mode="max",
            every_n_train_steps=500,
            dirpath=args.log_dir,
            save_top_k=1,
        ),
    ]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="valid_loss", mode="min"))

    # the following trainer serves only one purpose, that is finding the cyclic learning rate because
    # it is not supported on ddp
    print("creating the first trainer")
    disposable_trainer = Trainer(
        logger=False,
        auto_select_gpus=True,
        accelerator="gpu",
        devices=1,
        strategy="dp",
        num_nodes=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        auto_lr_find=True,
    )
    # the real one
    print("creating the second trainer")
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(
            find_unused_parameters=False, gradient_as_bucket_view=True
        ),
        max_epochs=args.epochs,
        callbacks=callbacks,
        auto_lr_find=False,  # will be manually set
    )

    # model instantiation
    # one can add a custom implementation to handle various learning techniques
    # i.e., regression, classification, rl, etc.
    arch = args.model_name
    lightning_model = modules.__dict__[f"MIPPLModel{arch}"](
        arch=arch,
        nheads=args.nheads,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        samples_path=args.samples_path,
        emb_size=args.emb_size,
    )

    lr_finder = disposable_trainer.tuner.lr_find(model=lightning_model, num_training=50)

    lr = lr_finder.suggestion()
    print(f"learning rate found after 200 iterations: {lr}")

    del disposable_trainer

    lightning_model.learning_rate = lr
    checkpoint = next(iter(glob.glob(f"{args.log_dir}/latest-*")), None)
    if checkpoint:
        print(f"found a checkpoint at {checkpoint}")

    print(f"fitting ...")
    trainer.fit(lightning_model, ckpt_path=checkpoint)


if __name__ == "__main__":
    model_names = sorted(
        name
        for name in modules.layers.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(modules.layers.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="AI4L Experiments Module")

    parser.add_argument("--log-dir", dest="log_dir", type=str, help="path to log dir")
    parser.add_argument("--name", dest="name", type=str, help="experiment name")
    parser.add_argument(
        "--model-name",
        dest="model_name",
        choices=["GCNN", "GraphSAGE", "GAT"],
        type=str,
        help="experiment name",
    )
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to be used")
    parser.add_argument("--cpus", default=1, type=int, help="Number of GPUs to be used")
    parser.add_argument(
        "--nheads", default=1, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--epochs", default=300, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--emb_size", default=1, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument("--seed", default=1, type=int, help="Number of GPUs to be used")
    parser.add_argument(
        "--learning_rate", default=1, type=float, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--samples-path",
        type=str,
        dest="samples_path",
        help="path to dump branch and bound samples",
    )
    args = parser.parse_args()

    main(args)
