from dataclasses import dataclass
from typing import Any

import torch
import yaml
from torch import Tensor


def max_strain_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> dict[str, float]:
    return {"max strain": float(node.value)}


class Chebyshev(yaml.YAMLObject):
    yaml_tag = "!Chebyshev"

    def __init__(self, coef: list[tuple[float, float]], scale: tuple[float, float]):
        self.coef = coef
        self.scale = scale


class AngleDispersive(yaml.YAMLObject):
    yaml_tag = "!AngleDispersive"

    def __init__(
        self,
        emission_lines,
        n_steps: int,
        two_theta_range: tuple[float, float],
        goniometer_radius_mm: float,
        sample_displacement_mu_m: tuple[float, float],
        caglioti: dict[str, tuple[float, float]],
        background: Chebyshev,
    ):
        self.emission_lines = emission_lines
        self.n_steps = n_steps
        self.two_theta_range = two_theta_range
        self.goniometer_radius_mm = goniometer_radius_mm
        self.sample_displacement_mu_m = sample_displacement_mu_m
        self.caglioti = caglioti
        self.background = background


class MaximumStrain(yaml.YAMLObject):
    yaml_tag = "!Maximum"

    def __init__(self, v: float):
        self.v = v


def denorm(x_norm: Tensor, vmin: float | Tensor, vmax: float | Tensor) -> Tensor:
    return x_norm * (vmax - vmin) + vmin


def normalize(x: Tensor, vmin: float | Tensor, vmax: float | Tensor) -> Tensor:
    d_is_zero = vmax - vmin == 0.0
    if isinstance(d_is_zero, bool):
        if d_is_zero:
            return x
    else:
        if d_is_zero.any():
            return x

    return (x - vmin) / (vmax - vmin)


@dataclass
class CfgExtrema:
    bkg_coef: list[tuple[float, float]]
    bkg_scale: tuple[float, float]
    sd: tuple[float, float]
    u: tuple[float, float]
    v: tuple[float, float]
    w: tuple[float, float]

    max_strain: list[float]
    ds_nm: list[tuple[float, float]]
    # mustrain: list[tuple[float, float]]

    def combine_normalize(
        self,
        strains: Tensor,
        mean_ds_nm: Tensor,
        # mustrain: Tensor,
        bkg_params: Tensor,
        sample_displacement_mu_m: Tensor,
        instrument_parameters: Tensor,
        ds_etas: Tensor,
        # mustrain_etas: Tensor,
        device: str | torch.device,
    ):
        strain_ident = torch.tensor([1, 0, 1, 0, 0, 1], device=device).float()
        for i, strain_amplitude in enumerate(self.max_strain):
            strains[:, i] -= strain_ident
            strains[:, i] = normalize(
                strains[:, i], -strain_amplitude, strain_amplitude
            )

        for i, (ds_min, ds_max) in enumerate(self.ds_nm):
            mean_ds_nm[:, i] = normalize(mean_ds_nm[:, i], ds_min, ds_max)

        sd = normalize(sample_displacement_mu_m[:, None], *self.sd)

        # bkg_scale
        bkg_params[:, 0] = normalize(bkg_params[:, 0], *self.bkg_scale)

        # chebyshev coefficients
        for i, c_range in enumerate(self.bkg_coef):
            bkg_params[:, i + 1] = normalize(bkg_params[:, i + 1], *c_range)

        cag_lo = torch.tensor([self.u[0], self.v[0], self.w[0]], device=device)
        cag_hi = torch.tensor([self.u[1], self.v[1], self.w[1]], device=device)
        caglioti = normalize(instrument_parameters[:, :3], cag_lo, cag_hi)

        per_pattern = torch.cat([caglioti, sd, bkg_params], dim=1)
        per_phase = torch.cat(
            [
                strains,                       # 6 | 0
                mean_ds_nm[:, :, None],        # 1 | 6
                ds_etas[:, :, None],           # 1 | 7
                # mustrain[:, :, None],          # 1 | 8
                # mustrain_etas[:, :, None],     # 1 | 9 
            ],
            dim=2,
        )

        return per_pattern, per_phase

    def denormalize(
        self, per_pattern: Tensor, per_phase: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        strain_ident = torch.tensor([1, 0, 1, 0, 0, 1], device=per_phase.device).float()
        strains = torch.zeros_like(per_phase[:, :, :6])
        for i, strain_amplitude in enumerate(self.max_strain):
            strains[:, i] = denorm(
                per_phase[:, i, :6], -strain_amplitude, strain_amplitude
            )
            strains[:, i] += strain_ident

        mean_ds_nm = torch.zeros_like(per_phase[:, :, 6])
        for i, (d0, d1) in enumerate(self.ds_nm):
            mean_ds_nm[:, i] = denorm(per_phase[:, i, 6], d0, d1)

        ds_etas = per_phase[:, :, 7]
        # mustrain = torch.zeros_like(per_phase[:, :, 8])
        # for i, (s0, s1) in enumerate(self.mustrain):
        #     mustrain[:, i] = denorm(per_phase[:, i, 8], s0, s1)

        # mustrain_etas = per_phase[:, :, 9]

        cag_lo = torch.tensor(
            [self.u[0], self.v[0], self.w[0]], device=per_pattern.device
        ).unsqueeze(0)
        cag_hi = torch.tensor(
            [self.u[1], self.v[1], self.w[1]], device=per_pattern.device
        ).unsqueeze(0)
        caglioti = denorm(per_pattern[:, :3], cag_lo, cag_hi)

        sd = denorm(per_pattern[:, 3].squeeze(), *self.sd)
        bkg_scale = denorm(per_pattern[:, 4].squeeze(), *self.bkg_scale)
        bkg_coef = torch.zeros(per_pattern[:, 5:].shape, dtype=torch.float32)
        # chebyshev coefficients
        for i, c_range in enumerate(self.bkg_coef):
            bkg_coef[:, i] = denorm(per_pattern[:, 5 + i], *c_range)

        return (
            strains,
            mean_ds_nm,
            ds_etas,
            # mustrain,
            # mustrain_etas,
            bkg_coef,
            bkg_scale,
            caglioti,
            sd,
        )


def load_config_extrema(path: str) -> CfgExtrema:
    with open(path, "r") as file:
        l = yaml.UnsafeLoader
        l.add_constructor("!Maximum", max_strain_constructor)
        d = yaml.load(file, l)

    max_strain = []
    ds = []
    # mustrain = []
    for s in d["sample_parameters"]["structures"]:
        max_strain.append(s["strain"]["max strain"])
        ds.append(s["mean_ds_nm"])
        # mustrain.append(s["mustrain"]["amplitude"])


    try:
        caglioti = d["kind"].caglioti
        (
            u,
            v,
            w,
        ) = (
            caglioti["u"],
            caglioti["v"],
            caglioti["w"],
        )
    except AttributeError:
        u, v, w = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    return CfgExtrema(
        bkg_coef=d["kind"].background.coefs,
        bkg_scale=d["kind"].background.scale,
        max_strain=max_strain,
        sd=d["kind"].sample_displacement_mu_m,
        ds_nm=ds,
        u=u,
        v=v,
        w=w,
        # mustrain=mustrain,
    )


if __name__ == "__main__":
    print(load_config_extrema("./data-val.yml"))
