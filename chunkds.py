import os
import zipfile
from functools import partial
from multiprocessing.sharedctypes import RawArray

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

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
