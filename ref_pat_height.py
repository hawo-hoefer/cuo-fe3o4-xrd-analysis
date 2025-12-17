import os
import subprocess as sp
import sys

import numpy as np


def get_ref_pat_height(print_results: bool = True) -> tuple[float, float]:
    workdir = os.path.join(os.path.dirname(__file__), "data", "ref_pattern")
    ret = sp.run(
        ["yaxs", "ref.yaml", "--overwrite", "-o", "ref"],
        cwd=workdir,
        capture_output=True,
    )
    if ret.returncode != 0:
        print("ERROR: Could not generate reference pattern. Quitting", file=sys.stderr)
        print("yaxs stdout:")
        print(ret.stdout.decode("utf-8"))
        print("========================")
        print("yaxs stderr:")
        print(ret.stderr.decode("utf-8"))
        exit(1)

    ret = sp.run(
        ["yaxs", "ref.yaml", "--display-hkls=structure"],
        cwd=workdir,
        capture_output=True,
    )
    if ret.returncode != 0:
        print(
            "ERROR: Could not generate reference peak structure height. Quitting",
            file=sys.stderr,
        )
        print("yaxs stdout:")
        print(ret.stdout.decode("utf-8"))
        print("========================")
        print("yaxs stderr:")
        print(ret.stderr.decode("utf-8"))
        exit(1)

    stderr = ret.stderr.decode("utf-8")
    i_hkls = []
    for line in stderr.splitlines():
        if not "i_hkl" in line:
            continue
        i_hkl = float(line.split("i_hkl:")[1].split("d_hkl")[0].strip())
        i_hkls.append(i_hkl)

    generated_dataset = np.load(os.path.join(workdir, "ref", "data.npz"))
    max_intens = generated_dataset["intensities"].max()
    max_ihkl = max(i_hkls)
    if print_results:
        print(f"Reference pattern height: {max_intens:.2f}")
        print(f"Reference maximum F_hkl:  {max_ihkl:.2f}")
    return max_intens, max_ihkl


if __name__ == "__main__":
    get_ref_pat_height(print_results=True)
