from multiprocessing import cpu_count
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from datasets._interface import GraphDataset
from torch_geometric.loader import DataLoader

from modules.layers import GCNN

from ._utils import BBAccuracy


class MIPPLModelGCNN(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        learning_rate: float,
        samples_path: str,
        batch_size: int,
    ) -> None:
        super(MIPPLModelGCNN, self).__init__()

        self.loss_function = lambda logits, targets: F.cross_entropy(
            logits, targets, reduction="mean"
        )
        self.learning_rate = learning_rate
        self.sample_files = [
            str(path) for path in Path(samples_path).glob("sample_*.pkl")
        ]

        self.model = GCNN()
        self.batch_size = batch_size

        self.train_acc = BBAccuracy()
        self.valid_acc = BBAccuracy()
        self.save_hyperparameters()

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        return self.model(
            constraint_features, edge_indices, edge_features, variable_features
        )

    def _pad_tensor(self, input_, pad_sizes, pad_value=-1e8):
        max_pad_size = pad_sizes.max()
        output = input_.split(pad_sizes.cpu().numpy().tolist())
        output = torch.stack(
            [
                F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
                for slice_ in output
            ],
            dim=0,
        )
        return output

    def training_step(self, batch, _):
        batch = batch.to(self.device)
        logits = self(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        logits = self._pad_tensor(logits[batch.candidates], batch.nb_candidates)
        loss = F.cross_entropy(logits, batch.candidate_choices, reduction="mean")

        true_scores = self._pad_tensor(batch.candidate_scores, batch.nb_candidates)
        true_bestscore = true_scores.max(dim=-1, keepdims=True).values

        predicted_bestindex = logits.max(dim=-1, keepdims=True).indices

        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        self.train_acc(
            predicted_bestindex, true_scores, true_bestscore, batch.num_graphs
        )
        return loss

    def training_epoch_end(self, _):
        self.log("train_acc", self.train_acc.compute())

    def validation_step(self, batch, _):
        batch = batch.to(self.device)
        logits = self(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        logits = self._pad_tensor(logits[batch.candidates], batch.nb_candidates)
        loss = F.cross_entropy(logits, batch.candidate_choices, reduction="mean")

        true_scores = self._pad_tensor(batch.candidate_scores, batch.nb_candidates)
        true_bestscore = true_scores.max(dim=-1, keepdims=True).values

        predicted_bestindex = logits.max(dim=-1, keepdims=True).indices

        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True)
        self.valid_acc(
            predicted_bestindex, true_scores, true_bestscore, batch.num_graphs
        )

    def validation_epoch_end(self, _):
        self.log("valid_acc", self.valid_acc.compute())

    def predict_step(self, batch, _):
        logits = self(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)

    def setup(self, stage) -> None:
        train_files = self.sample_files[: int(0.8 * len(self.sample_files))]
        valid_files = self.sample_files[int(0.8 * len(self.sample_files)) :]

        self.train_dataset: GraphDataset = GraphDataset(train_files)
        self.val_dataset: GraphDataset = GraphDataset(valid_files)

    def train_dataloader(self) -> DataLoader:
        train_loader = torch_geometric.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=3,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = torch_geometric.loader.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
        )
        return val_loader
