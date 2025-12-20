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
import pandas as pd
import json
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from chunkds import ChunkDS
from train import add_noise
import torch

# %%
kit_green = np.array([0, 150, 130]) / 255
kit_blue = np.array([70, 100, 170]) / 255
kit_black = np.array([0, 0, 0]) / 255
kit_black70 = np.array([64, 64, 64]) / 255

kit_yellow = np.array([252, 229, 0]) / 255
kit_orange = np.array([223, 155, 27]) / 255
kit_maygreen = np.array([140, 182, 60]) / 255
kit_red = np.array([162, 34, 35]) / 255
kit_purple = np.array([163, 16, 124]) / 255
kit_brown = np.array([167, 130, 46]) / 255
kit_cyan = np.array([35, 161, 224]) / 255

# %%
matplotlib.rc("text", usetex=False)
matplotlib.rc("font", size=7)

# %%
cmap = sns.color_palette("crest", as_cmap=True)

# %%
with open("./err_meta.json", "r") as file:
    err_meta = json.load(file)

errors = np.load("./errors.npy")
meta = pd.DataFrame(err_meta, columns=["radiation", "noise"])


# %%
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

def add_noise_exact(inputs, noise_amt: float):
    X = inputs[0][0]
    C = inputs[1][6]
    X += torch.randn(size=X.shape, device=X.device) * noise_amt
    return X, C


# %%
ds = ChunkDS("./data/cuka/test/")

# %%
examples = []
for noise_amt in [-6, -5, -4, -3, -2]:
    noise_amt = 2**noise_amt * 420
    
    X, C = add_noise_exact(ds[0:1000], noise_amt=noise_amt)
    X_noise = X[(C[:, 0] - 0.5).abs() < 0.01]
    examples.append(X_noise)

# %%
fig = plt.figure(figsize=(3.29, 3.29 / 16 * 9), layout="constrained")
thetas = np.linspace(10, 70, 2048)
idx = 15
plt.plot(thetas, examples[-1][idx], color=kit_blue, label=r"$\sigma = 2^{-2} h_\mathrm{ref}$")
plt.plot(thetas, examples[0][idx], color=kit_orange, label=r"$\sigma = 2^{-6} h_\mathrm{ref}$")
plt.xlabel(r"$2\theta\ [\mathrm{deg}]$")
plt.ylabel("I [a.u.]")
plt.yticks([])
plt.legend(loc="lower right")
plt.savefig("synthetic_data_example.pdf", bbox_inches="tight")

# %%
fig = plt.figure(figsize=(3.29, 3.29 / 16 * 9), layout="constrained")

label_map = {
    "cuka1": r"$\mathrm{CuK}\alpha_1$",
    "cuka": r"$\mathrm{CuK}\alpha_{1/2}$",
    "cukab": r"$\mathrm{CuK}\alpha_{1/2}/\beta$",
    "c3": r"$\mathcal{C}_3$",
    "c4": r"$\mathcal{C}_4$"
}

colors = [kit_blue, kit_orange, kit_maygreen, kit_cyan, kit_purple]
for i, radiation in enumerate(meta["radiation"].unique()):
    meta_subset = meta.query(f"radiation == '{radiation}'")
    N = meta_subset.shape[0]
    p = {"color": colors[i]}
    label = label_map[radiation]
    
    plt.boxplot(1 - errors[meta_subset.index].T,
                conf_intervals=[(0.025, 0.975) for _ in range(N)],
                positions=np.log2(meta_subset["noise"]) + i * 0.15 - (0.15 * 5 / 2) + 0.15 / 2,
                widths=[0.1 for _ in range(N)],
                showfliers=False,
                boxprops=p,
                whiskerprops=p,
                capprops=p,
                medianprops=p,
                label=label
               )

noise_levels = [-6, -5, -4, -3, -2]

for level, example in zip(noise_levels, examples):
    w = 512
    off = 725
    plt.plot(np.linspace(level - 0.45, level + 0.45, w), (normalize(example[idx][off:off+w]) - 0.5) * 0.025 + 1.025,
             lw=1, color="black")

plt.xticks(noise_levels, [f"$2^{{{l}}}$" for l in noise_levels])
plt.xlabel(r"$\sigma_\text{max} / h_\mathrm{ref}$")
plt.ylabel(r"$\mathcal{Q}$")
plt.ylim(0.78, 1.05)
plt.legend(loc="lower left", ncol=2)
plt.savefig("./synthetic_data_noise_tolerance.pdf", bbox_inches="tight")
