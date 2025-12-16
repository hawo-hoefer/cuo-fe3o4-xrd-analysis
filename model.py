import torch
from torch import Tensor, nn

from model_util import (
    conv_block,
    fc_block,
    last_out_channels,
    layer_from_cfg,
    size_after,
)


class GlobalMaxPool(nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        return X.max(dim=self.dim).values

def denorm_arr(X: Tensor, bounds: Tensor):
    return X * (bounds[1] - bounds[0]) + bounds[0]


def denormalize_model_outputs(
    outputs: tuple[Tensor, Tensor, Tensor],
    strain_range: Tensor,
    cs_range: Tensor,
    eta_range: Tensor,
    cag_range: Tensor,
    sd_range: Tensor,
) -> dict[str, Tensor]:
    composition, per_phase, per_pattern = outputs

    vals = {
        "crystallite_sz_nm": (per_phase[:, :, 0], cs_range),
        "eta": (per_phase[:, :, 1], eta_range),
        "strain": (per_phase[:, :, 2:], strain_range),
        "cag_uvw": (per_pattern[:, :3], cag_range),
        "sample_displacement_mu_m": (per_pattern[:, 3], sd_range),
    }

    ret: dict[str, Tensor] = {}
    for k, (v, bounds) in vals.items():
        ret[k] = denorm_arr(v, bounds)

    ret["composition"] = composition
    return ret


class ConvRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_phases: int,
        n_per_phase: int,
        n_per_pattern: int,
        f: int = 1,
    ) -> None:
        """Create a MultiScaleRegressor model

        Args:
            input_size: input pattern size
            n_phases: number of output phases
            n_per_phase: number of outputs per phase
            n_per_pattern: number of outputs per pattern (excluding those per phase)
            f: scale factor for kernel size in feature extractor and channels in reduce config
        """
        super().__init__()

        self.n_phases = n_phases
        self.n_per_phase = n_per_phase
        self.n_per_pattern = n_per_pattern

        # fmt: off
        only_convs = [
            conv_block(1,   8,   3, pool=2, stride=1, act="leaky_relu", padding=0),
            conv_block(8,   16,  3, pool=1, stride=1, act="leaky_relu", padding=0),
            conv_block(16,  32,  3, pool=1, stride=1, act="leaky_relu", padding=0),
            conv_block(32,  64,  3, pool=1, stride=1, act="leaky_relu", padding=0),
        ]
        # fmt: on

        feat = n_phases * (n_per_phase + 1) + n_per_pattern

        self.F = nn.Sequential(
            *[
                nn.Sequential(*[layer_from_cfg(**c) for c in block])
                for block in only_convs
            ],
            nn.LazyLinear(386),
            GlobalMaxPool(dim=1),
            nn.Dropout(p=0.5),
            nn.LazyLinear(feat),
            nn.LeakyReLU(),
            nn.LazyLinear(feat),
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()

        # self.intermediate = nn.Sequential(
        #     nn.LazyLinear(int((self.n_phases + self.n_phases * n_per_phase + self.n_per_pattern) * f)),
        #     nn.GELU(),
        # )

        # fmt: off
        self.composition = nn.Sequential(
            nn.LazyLinear(self.n_phases),
            nn.Softmax(1),
        )

        self.per_phase = nn.Sequential(
            # per-class meta info: eta, crystallite_sz_nm -> 2 outputs
            # nn.Linear(mid_size, self.n_phases * n_per_phase),
            nn.LazyLinear(self.n_phases * self.n_per_phase),
            nn.Sigmoid(), 
        )
        self.per_pattern = nn.Sequential(
            # per-pattern meta info: cagliotti_params, sample_displacement, bkg_scale, cheby_coefs
            # for now, we don't output background info -> maybe later
            # nn.Linear(mid_size, n_per_pattern),
            nn.LazyLinear(n_per_pattern),
            nn.Sigmoid()
        )
        # fmt: on

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = x.unsqueeze(1)

        x = self.F(x)
        # x = self.intermediate(x)

        composition = self.composition(x)
        per_phase = self.per_phase(x).reshape(-1, self.n_phases, self.n_per_phase)
        per_pattern = self.per_pattern(x)

        return composition, per_phase, per_pattern


class MultiScaleRegressor(nn.Module):
    def __init__(self, input_size: int, n_phases: int, n_per_phase: int, n_per_pattern: int, f: int = 1, of: int = 1) -> None:
        """Create a MultiScaleRegressor model

        Args:
            input_size: input pattern size
            n_phases: number of output phases
            n_per_phase: number of outputs per phase
            n_per_pattern: number of outputs per pattern (excluding those per phase)
            f: scale factor for kernel size in feature extractor and channels in reduce config
        """
        super().__init__()

        # fmt: off
        conv_cfgs = [
            conv_block(       1, 2**1 * f, 2**4 + 1, pool=3, stride=1),
            conv_block(2**1 * f, 2**2 * f, 2**3 + 1, pool=3, stride=1),
            conv_block(2**2 * f, 2**3 * f, 2**2 + 1, pool=3, stride=1),
            conv_block(2**3 * f, 2**4 * f, 2**1 + 1, pool=3, stride=1),
        ]
        # fmt: on

        self.n_phases = n_phases
        self.n_per_phase = n_per_phase
        self.n_per_pattern = n_per_pattern

        self.conv = nn.ModuleList(
            [
                nn.Sequential(*[layer_from_cfg(**c) for c in conv_cfg])
                for conv_cfg in conv_cfgs
            ]
        )

        ol = input_size
        ols = []
        ocs = []
        for cfg in conv_cfgs:
            ol = size_after(ol, cfg)
            ols.append(ol)
            ocs.append(last_out_channels(cfg))

        # fmt: off
        reduce_cfg = [
            conv_block(ocs[0], self.n_phases * of, 2**4 + 1, pool=27, stride=1),
            conv_block(ocs[1], self.n_phases * of, 2**3 + 1, pool=9,  stride=1),
            conv_block(ocs[2], self.n_phases * of, 2**2 + 1, pool=3,  stride=1),
            conv_block(ocs[3], self.n_phases * of, 2**1 + 1, pool=0,  stride=1),
        ]
        # fmt: on

        for i, cfg in enumerate(reduce_cfg):
            ol = size_after(ols[i], cfg)

        out_size = 0
        red_sizes = []
        for i, cfg in enumerate(reduce_cfg):
            size = size_after(ols[i], cfg)
            red_sizes.append(size)
            channels = last_out_channels(cfg)
            out_size += size * channels


        self.reducers = nn.ModuleList(
            [
                nn.Sequential(*[layer_from_cfg(**c) for c in cfg], 
                              nn.BatchNorm1d(self.n_phases * of),
                              nn.Dropout(p=0.5),
                              nn.Flatten())
                for cfg, rs in zip(reduce_cfg, red_sizes)
            ]
        )

        n_outputs_flat = (self.n_per_phase + 1) * self.n_phases + self.n_per_pattern
        mid_size = of * n_outputs_flat
        self.intermediate = nn.Sequential(
            nn.Linear(out_size, mid_size),
            nn.LeakyReLU(),
        )


        # fmt: off
        self.composition = nn.Sequential(
            nn.Linear(mid_size, self.n_phases),
            nn.Softmax(1),
        )

        self.per_phase = nn.Sequential(
            # per-class meta info: eta, crystallite_sz_nm -> 2 outputs
            nn.Linear(mid_size, self.n_phases * n_per_phase),
            nn.Sigmoid(), 
        )
        self.per_pattern = nn.Sequential(
            # per-pattern meta info: cagliotti_params, sample_displacement, (, bkg_scale, cheby_coefs)
            # for now, we don't output background info -> maybe later
            nn.Linear(mid_size, n_per_pattern),
            nn.Sigmoid()
        )
        # fmt: on

        with torch.no_grad():
            xs = [torch.rand(size=[10, 1, input_size])]
            for i, c in enumerate(self.conv):
                xs.append(c(xs[i]))

            for i in range(1, len(xs)):
                xs[i] = self.reducers[i - 1](xs[i])
        self.concat_shape = torch.concatenate(xs[1:], dim=1).shape[1]
        self.reduced_shapes = [x.shape[-1] for x in xs[1:]]


    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        concat = torch.empty(x.shape[0], self.concat_shape)
        start = 0
        for i, c in enumerate(self.conv):
            x = c(x)
            size = self.reduced_shapes[i]
            concat[:, start:start+size] = self.reducers[i](x)
            start += size

        x = self.intermediate(concat)

        composition = self.composition(x)
        per_phase = self.per_phase(x).reshape(-1, self.n_phases, self.n_per_phase)
        per_pattern = self.per_pattern(x)
        return composition, per_phase, per_pattern
