import typing
from typing import Callable, Literal

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torch import Tensor, is_grad_enabled



class Metrics:
    def __init__(self, dev: str, **funcs: Callable[[list[Tensor], list[Tensor]], Tensor]):
        self.acc: dict[str, list[Tensor]] = {name: [] for name in funcs.keys()}
        self.funcs = funcs
        self.dev = dev
        self.n: list[int] = []

    def reset(self):
        self.n: list[int] = []
        for k in self.acc:
            self.acc[k] = []

    def update(self, y: list[Tensor], Y: list[Tensor]) -> dict[str, Tensor]:
        self.n.append(Y[0].shape[0])
        for k in self.acc:
            self.acc[k].append(self.funcs[k](y, Y))

        return {k: self.acc[k][-1] for k in self.acc.keys()}
    
    
    def finalize(self) -> dict[str, Tensor]:
        total_samples = sum(self.n)
        ret = {
            k: sum((a * float(n) for a, n in zip(self.acc[k], self.n)), torch.tensor(0.0, device=self.dev)) / total_samples for k in self.acc.keys()
        }

        return ret


Reduction = Literal['mean', 'sum', 'none']
class TVDLoss(torch.nn.Module):
    """XRD Phase fraction identification loss (total variation distance).

    For models where the output is passed through a softmax activation, this loss computes the probability mass which needs to be moved in order to achieve the desired solution.
    This assumes that the model output and the target sum to 1.
    This is the jensen shannon divergence / total variation distance:
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        assert reduction in typing.get_args(Reduction)
        self.reduction: Reduction = reduction


    def forward(self, model_output: list[Tensor], target: list[Tensor]):
        diff = 1 / 2 * torch.abs(model_output[0] - target[0])
        if self.reduction == 'mean':
            return diff.sum(dim=1).mean()
        elif self.reduction == "sum":
            return diff.sum(dim=1).sum()
        elif self.reduction == "none":
            return diff

        assert False, "unreachable"


class MetadataScore(torch.nn.Module):
    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        assert reduction in typing.get_args(Reduction)
        self.reduction: Reduction = reduction
        self.mae = L1Loss(reduction="none")


    def forward(self, input: list[Tensor], target: list[Tensor]):
        by_phase = self.mae(input[1], target[1])
        by_pattern = self.mae(input[2], target[2])

        n_bkg_coefs = by_pattern.shape[1] - 4

        # weight the by-phase errors using the composition, so that 
        # it doesn't matter if properties of nonexistent phases are
        # predicted wrongly
        composition = input[0]
        by_phase *= composition.unsqueeze(-1) * composition.shape[-1]
        # by_phase /= by_phase.shape[2]
        # by_pattern /= by_pattern.shape[1]
        # make background amplitude as important as the parameters
        # by_pattern[:, 4] *= n_bkg_coefs
        # by_pattern[:, 4:] /= n_bkg_coefs * 2


        if self.reduction == "sum":
            by_phase = by_phase.sum()
            by_pattern = by_pattern.sum()
        elif self.reduction == "mean":
            by_phase = by_phase.mean()
            by_pattern = by_pattern.mean()

        return (by_phase + by_pattern * 5) / 6



class TotalScore(torch.nn.Module):
    def __init__(self, meta_weight: float = 0.5, reduction: Reduction = "mean") -> None:
        super().__init__()
        assert reduction in typing.get_args(Reduction)
        self.reduction: Reduction = reduction
        # self.vf = MSELoss(reduction=reduction)
        self.vf = TVDLoss(reduction=reduction)
        self.meta_weight = meta_weight
        self.meta = MetadataScore(reduction=reduction)


    def forward(self, input: list[Tensor], target: list[Tensor]):
        vf = self.vf(input, target)
        meta = self.meta(input, target)

        return vf * (1 - self.meta_weight) + self.meta_weight * meta

class FirstComponentCE(torch.nn.Module):
    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        assert reduction in typing.get_args(Reduction)
        self.reduction: Reduction = reduction
        self.l = CrossEntropyLoss(reduction=reduction)

    def forward(self, input: list[Tensor], target: list[Tensor]):
         return self.l(input[0], target[0])



class ECE:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def __call__(self, y: Tensor, Y: Tensor) -> Tensor:
        """compute the expected calibration error

        Args:
            y: class probabilities
            Y: actual output classes

        Returns: mean expected calibration error over all classes
        """
        y = torch.nn.functional.softmax(y, dim=1)
        pred_labels = y.argmax(dim=1)

        ECE = torch.tensor(0.0, device=y.device)
        for l in range(y.shape[1]):
            probs = y[:, l]
            binary_label_true = Y == l
            binary_label_pred = pred_labels == l
            conf = y.max(dim=1).values

            for i in range(self.n_bins):
                p0 = i / (self.n_bins)
                p1 = (i + 1) / (self.n_bins)
                sel = (p0 < probs) & (probs <= p1)
                n_bm = sel.int().sum()


                if n_bm > 0:
                    confs_bin = conf[sel]
                    binary_label_true_bin = binary_label_true[sel]
                    binary_label_pred_bin = binary_label_pred[sel]
                    conf_b_m = confs_bin.mean()
                    acc_b_m = (binary_label_true_bin == binary_label_pred_bin).float().mean()
                    ECE += torch.abs(conf_b_m - acc_b_m) * n_bm / pred_labels.shape[0]

        ECE /= y.shape[1]
        return ECE

# ECE = \sum_{m = 1}^{M} \frac{|B_m|}{n}\left|\mathrm{acc}\ B_m - \mathrm{conf}\ B_m\right|



def accuracy(y: Tensor, Y: Tensor) -> Tensor:
    y = y.argmax(dim=1)
    a = (y == Y).sum() / y.shape[0]
    return a


def macro_avg_f1_score(y: Tensor, Y: Tensor) -> Tensor:
    y = y.argmax(dim=1)
    f1s = []
    for label in Y.unique():
        tp: Tensor = ((y == label) & (Y == label)).sum()
        fn: Tensor = ((y != label) & (Y == label)).sum()
        fp: Tensor = ((y == label) & (Y != label)).sum()

        f1 = 2 * tp / (2 * tp + fp + fn)
        f1s.append(f1)

    return sum(f1s, torch.tensor(0.0, device=y.device)) / len(f1s)

def micro_avg_f1_score(y: Tensor, Y: Tensor) -> Tensor:
    y = y.argmax(dim=1)
    micro_avg_f1 = torch.tensor(0.0, device=y.device)
    for label in Y.unique():
        tp: Tensor = ((y == label) & (Y == label)).sum()
        fn: Tensor = ((y != label) & (Y == label)).sum()
        fp: Tensor = ((y == label) & (Y != label)).sum()

        weight: Tensor = ((Y == label).sum() / Y.shape[0])
        f1 = 2 * tp / (2 * tp + fp + fn)
        micro_avg_f1 += weight * f1

    return micro_avg_f1

def bce(y: Tensor, Y: Tensor, label_smoothing: float = 0.00) -> Tensor:
    Y = torch.zeros_like(y).scatter_(1, Y[:,None], torch.ones_like(y))

    if label_smoothing:
        Y *= (1 - label_smoothing)
        Y[Y == 0] = label_smoothing / (Y.shape[-1] - 1)

    l = torch.nn.functional.binary_cross_entropy(y.squeeze(), Y)
    return l

def calc_num_hits(y: Tensor, Y: Tensor) -> Tensor:
    return (torch.argmax(Y, dim=1) == torch.argmax(y, dim=1)).to(torch.float32).sum()


def r2(y: Tensor, Y: Tensor):
    tot = (Y - Y.mean(dim=0)).square().sum(dim=1)
    res = (y - Y).square().sum(dim=1)
