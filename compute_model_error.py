import json
import os
from functools import partial
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from parse_cfg import load_config_extrema
from ref_pat_height import get_ref_pat_height
from train import ChunkDS, add_noise, emission_lines, noises

workdir = os.path.dirname(__file__)
test_save_dir = os.path.join(workdir, "trained_models")

ref_pat_height, _ = get_ref_pat_height(print_results=False)

all_errs = []
for line in emission_lines:
    test = ChunkDS(os.path.join(workdir, "data", line, "test"), num_threads=5)
    data_cfg_path = os.path.join(workdir, "data", line, "train.yaml")
    cex = load_config_extrema(data_cfg_path)

    for noise in noises:
        noise_amt = ref_pat_height * noise
        dl_cfg = {
            "batch_size": 8192,
            "collate_fn": partial(add_noise, noise_amt=noise_amt, cex=cex),
            "prefetch_factor": 4,
            "pin_memory": True,
            "shuffle": True,
            "persistent_workers": False,
            "num_workers": 24,
            "generator": torch.Generator(device="cuda").manual_seed(1234),
        }
        dl = DataLoader(test, **dl_cfg)
        model_path = os.path.join(test_save_dir, line, str(noise), "best_model.pt")
        model = torch.load(model_path, weights_only=False).cuda().eval()
        tvds = []
        for x, (composition, per_phase, per_pattern) in dl:
            composition = composition.cuda()
            with torch.no_grad():
                c_, phase_, pat_ = model(x.cuda())
                tvd = ((composition - c_).abs().sum(dim=1) / 2).cpu()
                tvds.append(tvd)
        tvds = torch.cat(tvds).numpy()
        all_errs.append(tvds)

err_meta = [(line, noise) for line, noise in product(emission_lines, noises)]
with open("err_meta.json", "w") as file:
    json.dump(err_meta, file)

np.save("errors.npy", all_errs)
