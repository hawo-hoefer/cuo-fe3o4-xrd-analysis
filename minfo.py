import sys, os
import numpy as np
import torch
from model import ConvRegressor, MultiScaleRegressor

shapes = {}
def get_hook(name):
    def hook(model, input, output):
        print(output.shape[1:], name)

    return hook


def submods(module: torch.nn.Module, parent_path: str = "root"):
    name = module._get_name()
    children = list(module.children())
    if len(children) == 0:
        yield module, f"{parent_path}.{name}"

    for m in children:
        yield from submods(m, f"{parent_path}.{name}")

def psum(m):
    return sum(torch.numel(p) for p in m.parameters())


if __name__ == "__main__":
    # data_len = 1024
    # data_len = 2048
    try:
        d = np.load(os.path.join(os.path.dirname(__file__), "val", "data_0.npz"))
    except FileNotFoundError as e:
        d = np.load(os.path.join(os.path.dirname(__file__), "val", "data_00.npz"))
    data_len = d["intensities"].shape[-1]
    n_phases = d["volume_fractions"].shape[-1]
    # fmt: off
    #        strain + eta + domain_size
    n_per_phase = 6 + 1   + 1
    #              bkg + uvw + sd
    n_per_pattern = 14 + 3   + 1
    # fmt: on

    model = ConvRegressor(data_len, n_phases, n_per_phase, n_per_pattern, f=1)

    input = torch.rand(2, data_len)


    # for i, m in enumerate(model.reducers):
    #     m.register_forward_hook(get_hook(f"R{i}"))

    # model(input)
    # print(f"Out: {sum([torch.numel(p) for p in model.out.parameters()]):,}")
    # print(f"Red: {sum([torch.numel(p) for p in model.reducers.parameters()]):,}")
    # print(f"Con: {sum([torch.numel(p) for p in model.conv.parameters()]):,}")


    # print(f"Current Model: {sum([torch.numel(p) for p in model.parameters()]):,}")
    # print(f"FC layers: {sum([torch.numel(p) for p in model.out.parameters()]):,}")


    # model = BaselineRegressor(2048, 10)
    # print(f"Baseline: {sum([torch.numel(p) for p in model.parameters()]):,}")

    # model = ChunkMultiScaleRegressor(data_len, 8, 7, 3, 10)
    # model = ChunkMultiScaleRegressor(data_len, 4, 3, 8, 10)
    for elem, path in submods(model):
        elem.register_forward_hook(get_hook(path))

    comp, per_phase, per_pattern = model(input)

    n_conv = 0
    for elem, path in submods(model):
        if "Conv1d" in path:
            n_conv += sum(torch.numel(p) for p in elem.parameters())
    # print(sum(torch.numel(p) for n, p in model.named_parameters() if "Conv" in n))
    n_tot = psum(model)
    fc_params = psum(model.composition) + psum(model.per_phase) + psum(model.per_pattern)
    print(f"Tot:  {n_tot:,}")
    print(f"FC:   {n_tot - n_conv:,}")
    print(f"Down: {n_conv:,}")
    # print(f"Red:  {psum(model.reducers):,}")
    print("OUTPUT_SHAPES:")
    print(comp.shape, per_phase.shape, per_pattern.shape,
          "\n===============================================")
