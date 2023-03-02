import torch
from torchmetrics import Metric


class BBAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, target_bestscore, num_graphs
    ):
        accuracy = (target.gather(-1, preds) == target_bestscore).float().mean().item()

        self.correct += accuracy * num_graphs
        self.total += num_graphs

    def compute(self):
        return self.correct.float() / self.total
