from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from datasets._interface import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert

from modules.layers import GraphSAGE

from ._utils import BBAccuracy


class MIPPLModelGraphSAGE(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        nheads: int,
        learning_rate: float,
        samples_path: str,
        batch_size: int,
        emb_size: int,
    ) -> None:
        super(MIPPLModelGraphSAGE, self).__init__()

        self.learning_rate = learning_rate
        self.sample_files = [
            str(path) for path in Path(samples_path).glob("sample_*.pkl")
        ]

        self.model = GraphSAGE(
            nfeat=19, nhid=emb_size, nclass=1, dropout=0.0, alpha=0.2
        ).to(self.device)

        self.batch_size = batch_size

        self.train_acc = BBAccuracy()
        self.valid_acc = BBAccuracy()
        self.save_hyperparameters()

    def forward(self, features, edge_index, edge_attr):
        return self.model(features, edge_index, edge_attr)

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

    def _normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def _pad_adj(self, edge_index, edge_attr, num_vars, num_cons, fill=-9e15):
        cons_vars = edge_index
        vars_cons = edge_index[[1, 0]]
        cons_vars[0] = cons_vars[0] + num_vars
        vars_cons[1] = vars_cons[1] + num_vars

        values = torch.cat([edge_attr, edge_attr]).squeeze()
        indices = torch.cat([vars_cons, cons_vars], dim=1)
        return torch.sparse_coo_tensor(indices, values).coalesce()

    def training_step(self, batch, _):
        batch.to(self.device)
        constraint_features = F.pad(
            batch.constraint_features, (1, 13), "constant", 0
        )  # pad with zeros the amount of features that makes variable features and constraint features equal in size.
        features = torch.vstack([batch.variable_features, constraint_features])
        adj = self._pad_adj(
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            num_vars=batch.variable_features.shape[0],
            num_cons=batch.constraint_features.shape[0],
        ).to(self.device)
        logits = self(features, adj.indices(), adj.values()).squeeze(-1)
        logits = self._pad_tensor(logits[batch.candidates], batch.nb_candidates)
        loss = F.cross_entropy(logits, batch.candidate_choices)

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
        batch.to(self.device)
        constraint_features = F.pad(
            batch.constraint_features, (1, 13), "constant", 0
        )  # pad with zeros the amount of features that makes variable features and constraint features equal in size.
        features = torch.vstack([batch.variable_features, constraint_features])
        adj = self._pad_adj(
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            num_vars=batch.variable_features.shape[0],
            num_cons=batch.constraint_features.shape[0],
        ).to(self.device)
        logits = self(features, adj.indices(), adj.values()).squeeze(-1)
        logits = self._pad_tensor(logits[batch.candidates], batch.nb_candidates)
        loss = F.cross_entropy(logits, batch.candidate_choices)

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
        batch.to(self.device)
        constraint_features = F.pad(
            batch.constraint_features, (1, 13), "constant", 0
        )  # pad with zeros the amount of features that makes variable features and constraint features equal in size.
        features = torch.vstack([batch.variable_features, constraint_features])
        adj = convert.to_scipy_sparse_matrix(batch.edge_index, batch.edge_attr)
        logits = self(features, adj).squeeze(-1)
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
