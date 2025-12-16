import math
import os
import sys
import zipfile
from functools import partial
from multiprocessing.sharedctypes import RawArray
from time import monotonic_ns

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from metrics import Metrics, TotalScore, TVDLoss
from model import ConvRegressor, MultiScaleRegressor
from parse_cfg import load_config_extrema
from trainer import Trainer

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")


DSOutput = tuple[list[Tensor], list[Tensor]]


def load(
    offset: int,
    chunk_path: str,
    inputs: list[str],
    targets: list[str],
    inputs_buffers: list[NDArray],
    targets_buffers: list[NDArray],
):
    with np.load(chunk_path, mmap_mode="r") as data:
        for idx, input in enumerate(inputs):
            d = data[input]
            inputs_buffers[idx][offset : offset + d.shape[0]] = d
            del d

        for idx, target in enumerate(targets):
            d = data[target]
            targets_buffers[idx][offset : offset + d.shape[0]] = d
            del d


def get_joined_sizes(
    data_paths: list[str],
) -> tuple[dict[str, tuple[list[int], np.dtype]], list[int]]:
    size_info = {}
    chunk_sizes = []

    for chunk_path in data_paths:
        zf = zipfile.ZipFile(chunk_path, mode="r")
        arr_names = zf.namelist()
        n_samples = None
        for arr_name in arr_names:
            fp = zf.open(arr_name, "r")
            version = np.lib.format.read_magic(fp)

            if version[0] == 1:
                shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version[0] == 2:
                shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                print("File format not detected!")
                raise ValueError("Could not find file format in numpy array")
            fp.close()

            arr_name = arr_name.replace(".npy", "")

            if n_samples is None:
                n_samples = shape[0]
            else:
                if n_samples != shape[0]:
                    raise ValueError(
                        f"{chunk_path}: shape mismatch. All arrays must match in sample dimension. Got {shape[0]} but expected {n_samples}"
                    )

            if arr_name not in size_info:
                size_info[arr_name] = (shape, dtype)
            else:
                prev_shape, prev_dtype = size_info[arr_name]
                if prev_dtype != dtype:
                    raise ValueError(
                        f"{arr_name}: Type mismatch. Expected {prev_dtype}, got {dtype} when reading chunk {chunk_path}"
                    )
                assert len(prev_shape) == len(shape)
                joined_shape = []
                for i, (sp, si) in enumerate(zip(prev_shape, shape)):
                    if i == 0:
                        joined_shape.append(sp + si)
                        continue

                    if sp != si:
                        raise ValueError(
                            f"{arr_name}: Size mismatch in dimension {i}. Expected {sp}, got {si} when reading chunk {chunk_path}"
                        )
                    joined_shape.append(sp)

                size_info[arr_name] = (joined_shape, dtype)
        chunk_sizes.append(n_samples)
        zf.close()

    offsets = [0]
    for c in chunk_sizes[:-1]:
        offsets.append(offsets[-1] + c)

    return size_info, offsets


class ChunkDS(torch.utils.data.Dataset):
    """dataset of directory with meta.json and data separately

    Datasets can be created using `create_chunked_dataset`.
    These are then able to be read using `ChunkedPregeneratedMultiphaseDS`
    """

    def __init__(
        self, path: str, num_threads: int | None = None, device: str = "cpu"
    ) -> None:
        super().__init__()
        import json
        from typing import Any

        with open(os.path.join(path, "meta.json"), "r") as mf:
            meta: dict[str, Any] = json.load(mf)
            if meta.pop("chunked") != True:
                raise ValueError("Tried to open non-chunked dataset")
            data_paths = [os.path.join(path, f) for f in meta.pop("datafiles")]

        self.extra = meta.pop("extra") if "extra" in meta else None
        self.input_names = meta.pop("input_names")
        self.target_names = meta.pop("target_names")
        self.meta = meta
        self.device = device

        from tqdm.contrib.concurrent import thread_map

        size_info, offsets = get_joined_sizes(data_paths)

        inputs = []
        for input in self.input_names:
            shape, dtype = size_info[input]
            numel = 1
            for s in shape:
                numel *= s
            buf = RawArray(np.ctypeslib.as_ctypes_type(dtype), numel)
            inputs.append(np.asarray(buf, dtype=dtype).reshape(shape))

        targets = []
        for input in self.target_names:
            shape, dtype = size_info[input]
            numel = 1
            for s in shape:
                numel *= s
            buf = RawArray(np.ctypeslib.as_ctypes_type(dtype), numel)
            targets.append(np.asarray(buf, dtype=dtype).reshape(shape))

        lp = partial(
            load,
            inputs=self.input_names,
            targets=self.target_names,
            inputs_buffers=inputs,
            targets_buffers=targets,
        )
        thread_map(
            lp,
            offsets,
            data_paths,
            max_workers=num_threads,
            desc="loading dataset from disk",
        )
        self.inputs = inputs
        self.targets = targets

        # TODO: size verification

        self.pat_len = self.inputs[0].shape[-1]
        self.num_samples = self.inputs[0].shape[0]
        self.num_phases = self.targets[0].shape[1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> DSOutput:
        return [torch.tensor(input[i], device=self.device) for input in self.inputs], [
            torch.tensor(target[i], device=self.device) for target in self.targets
        ]

    def __getitems__(self, i: list[int]) -> DSOutput:
        return [torch.tensor(input[i], device=self.device) for input in self.inputs], [
            torch.tensor(target[i], device=self.device) for target in self.targets
        ]


TORCH_DTYPE = torch.float32

cex = load_config_extrema(os.path.join(os.path.dirname(__file__), "data-train.yaml"))


def add_noise(
    data: tuple[list[Tensor], list[Tensor]], noise_amt: float
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    (inputs,), (
        strains,
        instrument_parameters,
        mean_ds_nm,
        ds_etas,
        mustrain,
        mustrain_etas,
        volume_fractions,
        impurity_sum,
        sample_displacement_mu_m,
        impurity_max,
        weight_fractions,
        background_parameters,
    ) = data
    # TODO: maybe poisson noise
    X = inputs
    # X -= X.min(dim=-1, keepdim=True).values
    # rmin, rmax = 0.1, 5
    # roff_min, roff_max = 0, 1000
    # mul = torch.rand(size=[X.shape[0], 1], device=X.device) * (rmax - rmin) + rmin
    # roff = torch.rand(size=[X.shape[0], 1], device=X.device) * (roff_max - roff_min) + roff_min

    # X = torch.poisson(X * mul + roff)
    # X -= X.min(dim=-1, keepdim=True).values
    # X /= X.max(dim=-1, keepdim=True).values

    scale = torch.rand(size=[X.shape[0]], device=X.device) * noise_amt
    X += torch.randn(size=X.shape, device=X.device) * scale[:, None]
    X -= X.min(dim=-1, keepdim=True).values
    X /= X.max(dim=-1, keepdim=True).values

    per_pattern, per_phase = cex.combine_normalize(
        strains,
        mean_ds_nm,
        # mustrain,
        background_parameters,
        sample_displacement_mu_m,
        instrument_parameters,
        ds_etas,
        # mustrain_etas,
        X.device,
    )

    return X.to(TORCH_DTYPE), (
        volume_fractions.to(TORCH_DTYPE),
        per_phase.to(TORCH_DTYPE),
        per_pattern.to(TORCH_DTYPE),
    )


def train_model(train: ChunkDS, val: ChunkDS, seed: int, noise_amt: float):
    dl_cfg = {
        "batch_size": 8192,
        "collate_fn": partial(add_noise, noise_amt=noise_amt),
        "prefetch_factor": 4,
        "pin_memory": True,
        "shuffle": True,
        "persistent_workers": False,
        "num_workers": 24,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
    }
    torch.manual_seed(seed)
    train_loader = DataLoader(train, **dl_cfg)
    val_loader = DataLoader(val, **dl_cfg)

    X_, (composition, per_phase, per_pat) = next(iter(train_loader))
    print("got dummy batch")
    print("train", composition.shape, per_phase.shape, per_pat.shape)

    n_phases_ds = len(train.extra["encoding"])  # type: ignore
    assert n_phases_ds == per_phase.shape[1]

    model = MultiScaleRegressor(
        train.inputs[0].shape[-1],
        n_phases_ds,
        per_phase.shape[-1],
        per_pat.shape[-1],
        f=1,
        of=1,
    )

    _ = model(X_.cuda())

    lr = 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=lr, maximize=False, weight_decay=5e-3)

    sched = ReduceLROnPlateau(
        opt, threshold_mode="rel", patience=3, threshold=1e-2, factor=1e-1
    )
    # sched = StepLR(opt, 1, gamma=0.95)
    # sched = None
    # c_model = torch.compile(model)
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
        with_progress=True,
    )

    best_loss = 1e100
    print(
        f"training model with {sum(torch.numel(p) for p in model.parameters()):,} parameters"
    )

    epochs = 20
    save_path = "./trained.pt"

    l = open("train.log", "w")
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
    finally:
        l.close()
        del train_loader
        del val_loader
        del train
        del val

    return save_path


def main():
    emission_lines = ["cuka1", "cuka", "cukab", "c3", "c4"]

    for line in emission_lines:
        train = ChunkDS(os.path.join("./data", line, "train"), num_threads=5)
        val = ChunkDS(os.path.join("./data", line, "val"), num_threads=5)
        for noise in [10, 20, 50, 100]:
            seed = 1234
            train_model(train, val, seed=seed, noise_amt=noise)


if __name__ == "__main__":
    main()
