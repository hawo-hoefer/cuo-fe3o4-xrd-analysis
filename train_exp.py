import os

from parse_cfg import load_config_extrema
from ref_pat_height import get_ref_pat_height
from train import train_model
from chunkds import ChunkDS

exp_base_dir = os.path.join(os.path.dirname(__file__), "exp_data_analysis")
save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "exp")

emission_lines = [("cuka", 0.05), ("cukab", 0.05)]

for line, noise in emission_lines:
    data_dir = os.path.join(exp_base_dir, line)
    train_cfg_path = os.path.join(data_dir, "train.yaml")
    train = ChunkDS(os.path.join(data_dir, "train"))
    val = ChunkDS(os.path.join(data_dir, "val"))
    ref_pat_height, _ = get_ref_pat_height(print_results=False)
    cex = load_config_extrema(train_cfg_path)
    line_save_dir = os.path.join(save_dir, line)

    if not os.path.exists(line_save_dir):
        os.makedirs(line_save_dir)

    best_model_path = os.path.join(line_save_dir, "best_model.pt")
    log_path = os.path.join(line_save_dir, "train.log")

    train_model(
        train,
        val,
        20,
        1234,
        noise * ref_pat_height,
        cex,
        best_model_path,
        log_path,
        with_progress=True,
        f=2,
        of=2,
        batch_size=8192,
        lr=1e-3,
    )
