# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
import pandas as pd
import json
import scipy.ndimage
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
import matplotlib as mpl
import seaborn as sns

# %%
mpl.rc('font', size=6.9)
kit_blue = np.array([70, 100, 170]) / 255
kit_orange = np.array([223, 155, 27]) / 255


# %%
def add_noise(x):
    scale = torch.ones(size=[x.shape[0]]) * 50
    x += torch.randn(size=x.shape, device=x.device) * scale[:, None]
    x -= x.min(dim=-1, keepdim=True).values
    x /= x.max(dim=-1, keepdim=True).values
    return x


# %%
with open("./real_data/meta.json", "r") as file:
    meta = json.load(file)

# %%
d = np.load("./real_data/data.npz")
df = pd.DataFrame(meta["samples"])
df["CuO"] = d["targets"][:, 0]
df["Fe3O4"] = d["targets"][:, 1]
filtered = df[df["filter"]]["idx"]
unfiltered = df[~df["filter"]]["idx"]
df.head()

# %%
filtered_ex_short = df.query("CuO == 0.5 and time_per_step_s == 0.1 and filter").iloc[0]["idx"]
unfiltered_ex_long = df.query("CuO == 0.5 and time_per_step_s == 5.0 and not filter").iloc[0]["idx"]

two_theta = np.linspace(10, 70, 2048)

fig = plt.figure(figsize=(3.29, 3.29 * 9 / 16))
plt.plot(two_theta, d["inputs"][filtered_ex_short], color=kit_blue, label=r"0.1s $\mathrm{CuK}\alpha_{1/2}$")
plt.plot(two_theta, d["inputs"][unfiltered_ex_long], color=kit_orange, label=r"5s $\mathrm{CuK}\alpha_{1/2}/\beta$")
plt.legend()
plt.yticks([])
plt.xlabel(r"$2\theta$ [deg]")
plt.ylabel(r"$\mathrm{I}\ [\mathrm{a.u.}]$")
plt.savefig("./example_exp_xrd.pdf", bbox_inches="tight")

# %%
m = torch.load("./trained_models/exp/cuka/best_model.pt", weights_only=False).eval().cpu()


targets = d["targets"][filtered]
inputs = d["inputs"][filtered]

with torch.no_grad():
    c, _, _ = m(torch.tensor(inputs).cpu().float())
    c = c.numpy()
    
tvd = np.abs(c[:, 0] - targets[:, 0]) / 2
q = 1 - tvd
time_per_step = df.set_index("idx").loc[filtered]["time_per_step_s"]
print(f"{q.min()=}")

mapp = ScalarMappable(norm=Normalize(vmin=time_per_step.min(), vmax=time_per_step.max()), cmap=sns.color_palette('crest', as_cmap=True))

fig, ax = plt.subplots(1, 2, figsize=(3.29, 3.29 / 16 * 9), layout="constrained")

ax[0].scatter(targets[:, 0], c[:, 0], marker="o", s=7, edgecolors=mapp.to_rgba(time_per_step), facecolor="#00000000")
ax[0].plot([0, 1], [0, 1], color="black", alpha=0.5, ls="--")
ax[0].set_xlabel("CuO weight fraction")
ax[0].set_ylabel("predicted CuO weight fraction")

ax[1].scatter(targets[:, 0], q, marker="o", s=7, edgecolors=mapp.to_rgba(time_per_step), facecolor="#00000000")
ax[1].axhline(q.mean(), label="mean", ls="dotted", color="black")
ax[1].legend(loc="lower right")
ax[1].set_xlabel("CuO weight fraction")
ax[1].set_ylabel(r"$\mathcal{Q}$")
ax[1].text(0.0, q.mean(), f"{q.mean().item():.3f}", ha="left", va="bottom")
plt.colorbar(mapp, ax=ax[1], label="time per step")
plt.savefig("experimental_results_cuka.pdf", bbox_inches="tight")

# %%
m = torch.load("./trained_models/exp/cukab/best_model.pt", weights_only=False).eval().cpu()

targets = d["targets"][unfiltered]
inputs = d["inputs"][unfiltered]

with torch.no_grad():
    c, _, _ = m(torch.tensor(inputs).cpu().float())
    c = c.numpy()


tvd = np.abs(c[:, 0] - targets[:, 0]) / 2
q = 1 - tvd
print(q.min())
time_per_step = df.set_index("idx").loc[unfiltered]["time_per_step_s"]
mapp = ScalarMappable(norm=Normalize(vmin=time_per_step.min(), vmax=time_per_step.max()), cmap=sns.color_palette('crest', as_cmap=True))

fig, ax = plt.subplots(1, 2, figsize=(3.29, 3.29 / 16 * 9), layout="constrained")

ax[0].scatter(targets[:, 0], c[:, 0], marker="o", s=7, edgecolors=mapp.to_rgba(time_per_step), facecolor="#00000000")
ax[0].plot([0, 1], [0, 1], color="black", alpha=0.5, ls="--")
ax[0].set_xlabel("CuO weight fraction")
ax[0].set_ylabel("predicted CuO weight fraction")

ax[1].scatter(targets[:, 0], q, marker="o", s=7, edgecolors=mapp.to_rgba(time_per_step), facecolor="#00000000")
ax[1].axhline(q.mean(), label="mean", ls="dotted", color="black")
ax[1].legend(loc="upper right")
ax[1].set_xlabel("CuO weight fraction")
ax[1].set_ylabel(r"$\mathcal{Q}$")
ax[1].text(0.0, q.mean(), f"{q.mean().item():.3f}", ha="left", va="bottom")
plt.colorbar(mapp, ax=ax[1], label="time per step")
plt.savefig("experimental_results_cukab.pdf", bbox_inches="tight")
