from collections.abc import Sequence
from typing import Literal
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from metrics import Metrics
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm


class Trainer:
    MetricDict = dict[str, Tensor]

    def __init__(
        self,
        metrics: Metrics,
        back_metric: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        with_progress: bool = False
    ):
        self.model = model
        self.m = metrics
        self.tl = train_loader
        self.vl = val_loader
        self.opt = optimizer
        self.schedulers = schedulers
        self.back_metric = back_metric
        self.with_progress = with_progress

    def half_epoch(self, p: None | tqdm, state: Literal["train", "val"]) -> MetricDict:
        self.m.reset()
        if state == "train":
            loader = self.tl
        elif state == "val":
            loader = self.vl
        else:
            raise ValueError(f"Invalid state for half_epoch: '{state}'. Must be either 'train' or 'val'.")

        for X, Y in loader:
            if isinstance(X, Sequence):
                X = [x.cuda() for x in X]
            else:
                X = [X.cuda()]


            if isinstance(Y, Sequence):
                Y = [y.cuda() for y in Y]
            else:
                Y = [Y.cuda()]

            pred = self.model(*X)
            l = self.m.update(pred, Y)[self.back_metric]
            if state == "train":
                self.opt.zero_grad()
                l.backward()
                self.opt.step()

            if p:
                p.set_description(f"{state}: {l.item():4.2e}")
                p.update()

        return self.m.finalize()

    def epoch(self) -> tuple[MetricDict, MetricDict]:
        self.model.train()
        if self.with_progress:
            p = tqdm(total=len(self.tl) + len(self.vl), leave=False)
        else:
            p = None

        tm = self.half_epoch(p, "train")
        self.model.eval()
        with torch.no_grad():
            vm = self.half_epoch(p, "val")

            for s in self.schedulers:
                if isinstance(s, ReduceLROnPlateau):
                    s.step(vm[self.back_metric])
                else:
                    s.step()

        return tm, vm
