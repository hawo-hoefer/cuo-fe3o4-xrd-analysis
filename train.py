import math
import os
import sys
from functools import partial
from time import monotonic_ns

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from chunkds import ChunkDS

from metrics import Metrics, TotalScore, TVDLoss
from model import ConvRegressor, MultiScaleRegressor
from parse_cfg import CfgExtrema, load_config_extrema
from ref_pat_height import get_ref_pat_height
from trainer import Trainer
import logging

logging.captureWarnings(True)

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")



TORCH_DTYPE = torch.float32
cex = None


def add_noise(
    data: tuple[list[Tensor], list[Tensor]], noise_amt: float, cex: CfgExtrema,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    (inputs,), (
        strains,
        instrument_parameters,
        mean_ds_nm,
        ds_etas,
        _mustrain, 
        _mustrain_etas,  # type: ignore
        volume_fractions,
        _impurity_sum,  # type: ignore
        sample_displacement_mu_m,
        _impurity_max,  # type: ignore
        _weight_fractions,  # type: ignore
        background_parameters,
    ) = data
    X = inputs

    scale = torch.rand(size=[X.shape[0]], device=X.device) * noise_amt

    X += torch.randn(size=X.shape, device=X.device) * scale[:, None]
    X -= X.min(dim=-1, keepdim=True).values
    X /= X.max(dim=-1, keepdim=True).values

    per_pattern, per_phase = cex.combine_normalize(
        strains,
        mean_ds_nm,
        background_parameters,
        sample_displacement_mu_m,
        instrument_parameters,
        ds_etas,
        X.device,
    )

    return X.to(TORCH_DTYPE), (
        volume_fractions.to(TORCH_DTYPE),
        per_phase.to(TORCH_DTYPE),
        per_pattern.to(TORCH_DTYPE),
    )


def train_model(
    train: ChunkDS,
    val: ChunkDS,
    epochs: int,
    seed: int,
    noise_amt: float,
    cfg_extrema: CfgExtrema,
    save_path: str,
    log_path: str,
    with_progress: bool,
    f: int = 2,
    of: int = 2,
    batch_size: int = 8192,
    lr: float = 1e-3,
):
    dl_cfg = {
        "batch_size": batch_size,
        "collate_fn": partial(add_noise, noise_amt=noise_amt, cex=cfg_extrema),
        "prefetch_factor": 2,
        "pin_memory": True,
        "shuffle": True,
        "persistent_workers": True,
        "num_workers": 12,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
    }
    torch.manual_seed(seed)
    train_loader = DataLoader(train, **dl_cfg)
    val_loader = DataLoader(val, **dl_cfg)

    X_, (_, per_phase, per_pat) = next(iter(train_loader))

    n_phases_ds = len(train.extra["encoding"])  # type: ignore
    assert n_phases_ds == per_phase.shape[1]

    model = MultiScaleRegressor(
        train.inputs[0].shape[-1],
        n_phases_ds,
        per_phase.shape[-1],
        per_pat.shape[-1],
        f=f,
        of=of,
    )

    _ = model(X_.cuda())

    opt = torch.optim.Adam(model.parameters(), lr=lr, maximize=False, weight_decay=5e-3)

    sched = ReduceLROnPlateau(
        opt, threshold_mode="rel", patience=3, threshold=1e-2, factor=1e-1
    )

    c_model = model
    m = Metrics("cuda", tvd=TVDLoss(), tot=TotalScore(meta_weight=0.05))
    t = Trainer(
        m,
        "tot",
        c_model,
        train_loader,
        val_loader,
        opt,
        [sched] if sched is not None else [],
        with_progress=with_progress,
    )

    best_loss = 1e100
    print(
        f"training model with {sum(torch.numel(p) for p in model.parameters()):,} parameters"
    )

    l = open(log_path, "w")
    try:
        title_str = f"{'epoch':5} {'time_ms':8} {'lr':8} {'t_tot':8} {'v_tot':8} {'t_tvd':8} {'v_tvd':8}\n"
        l.write(title_str)
        print(title_str, end="")
        for e in range(epochs):
            start = monotonic_ns()
            tm, vm = t.epoch()

            end = monotonic_ns()
            elapsed_ms = (end - start) / 1e6

            if best_loss > vm["tvd"]:
                best_loss = vm["tvd"]
                torch.save(model, save_path)

            lr = sched.get_last_lr()[0] if sched is not None else lr
            ostr = f"{e:5} {elapsed_ms:8.2e} {lr:8.2e} {tm['tot']:8.2e} {vm['tot']:8.2e} {tm['tvd']:8.2e} {vm['tvd']:8.2e}\n"

            l.write(ostr)
            print(ostr, end="")
            sys.stdout.flush()
            l.flush()
    except KeyboardInterrupt:
        print("Exiting due to KeyboardInterrupt")
        raise KeyboardInterrupt()
    finally:
        l.close()
        del train_loader
        del val_loader
        del train
        del val


emission_lines = ["cuka1", "cuka", "cukab", "c3", "c4"]
noises = [2**-6, 2**-5, 2**-4, 2**-3, 2**-2]

def main():
    global cex

    workdir = os.path.dirname(__file__)
    trained_models_dir = os.path.join(workdir, "trained_models")
    progress = True
    if "--no-progress" in sys.argv:
        progress = False

    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    ref_pat_height, _ = get_ref_pat_height(print_results=False)

    epochs = 20

    for line in emission_lines:
        print(f"================================================================")
        print(f"Loading data for emission lines {line}")
        print(f"================================================================")
        train = ChunkDS(os.path.join(workdir, "data", line, "train"), num_threads=5)
        val = ChunkDS(os.path.join(workdir, "data", line, "val"), num_threads=5)
        data_cfg_path = os.path.join(workdir, "data", line, "train.yaml")
        cex = load_config_extrema(data_cfg_path)

        for noise in noises:
            # noise as fraction of reference pattern height
            seed = 1234
            base_dir = os.path.join(trained_models_dir, line, f"{noise}")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            best_model_path = os.path.join(base_dir, "best_model.pt")
            log_path = os.path.join(base_dir, "train.log")
            print(f"================================================================")
            print(f"Training model for emission lines {line} and noise level {noise}")
            print(f"================================================================")
            try:
                train_model(
                    train,
                    val,
                    epochs,
                    seed,
                    noise * ref_pat_height,
                    cex,
                    best_model_path,
                    log_path,
                    progress,
                )
            except KeyboardInterrupt:
                exit(1)


if __name__ == "__main__":
    main()
