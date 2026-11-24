"""
plots.py - figurat per raportin, nga rezultatet e ruajtura (nuk ekzekuton simulatorin).

1. figures/convergence.png : fitness-i me i mire ne 30 skenare sipas numrit te vleresimeve,
                             per çdo run te fazes 1 (maskat) dhe fazes 2 (maskat + shtresa e daljes).
2. figures/masks_<run>.png : 4 maskat 11x11 te rover-it me te mire te nje run-i.
3. figures/mask_change_<run>.png : maskat e shembullit, maskat e evoluara dhe ndryshimi mes tyre.

Perdorimi (nga folderi kryesor):
    py analysis/plots.py
"""
import os
import csv
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt

RESULTS = "results"
FIGURES = "figures"
EXAMPLE_LAPTOP = 7.9844          
EXAMPLE_ROVER = os.path.join("data", "spoc2", "morphing", "example_rover.npy")

PHASES = {
    "Faza 1: maskat": ("masks_all_s*", "tab:blue"),
    "Faza 2: maskat + shtresa e daljes": ("output_s*", "tab:orange"),
}


def read_log(path):
    """Kthen (vleresimet, me_i_miri_ne_30) nga log.csv i nje run-i."""
    evals, best = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            evals.append(int(row["n_evals"]))
            best.append(float(row["best_so_far_30"]))
    return np.array(evals), np.array(best)


def plot_convergence():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(EXAMPLE_LAPTOP, color="gray", ls="--", lw=1, label="Rover-i shembull i ESA-s")
    for label, (pattern, color) in PHASES.items():
        runs = sorted(glob.glob(os.path.join(RESULTS, pattern, "log.csv")))
        for k, path in enumerate(runs):
            evals, best = read_log(path)
            start = EXAMPLE_LAPTOP
            cfg = os.path.join(os.path.dirname(path), "config.json")
            if os.path.exists(cfg):
                with open(cfg) as f:
                    start = json.load(f)["start_fitness_30"]
            evals = np.concatenate([[1], evals])
            best = np.concatenate([[start], best])
            ax.step(evals, best, where="post", color=color, alpha=0.8,
                    label=label if k == 0 else None)
    ax.set_xlabel("Numri i vleresimeve")
    ax.set_ylabel("Fitness-i me i mire ne 30 skenare (me i ulet = me mire)")
    ax.set_title("Konvergjenca e CMA-ES")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURES, "convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("U ruajt:", out)


def plot_masks(run_dir):
    path = os.path.join(run_dir, "best_rover.npy")
    if not os.path.exists(path):
        return
    masks = np.load(path)[:484].reshape(4, 11, 11)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    vmin, vmax = masks.min(), masks.max()
    for m, ax in enumerate(axes):
        im = ax.imshow(masks[m], cmap="terrain", vmin=vmin, vmax=vmax)
        ax.set_title(f"Forma {m + 1}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)
    name = os.path.basename(run_dir)
    fig.suptitle(f"Maskat e rover-it me te mire: {name}")
    out = os.path.join(FIGURES, f"masks_{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("U ruajt:", out)


def plot_mask_change(run_dir):
    """Rreshti 1: maskat e shembullit. Rreshti 2: maskat e evoluara. Rreshti 3: ndryshimi."""
    path = os.path.join(run_dir, "best_rover.npy")
    if not (os.path.exists(path) and os.path.exists(EXAMPLE_ROVER)):
        return
    before = np.load(EXAMPLE_ROVER)[:484].reshape(4, 11, 11)
    after = np.load(path)[:484].reshape(4, 11, 11)
    diff = after - before
    vmin, vmax = min(before.min(), after.min()), max(before.max(), after.max())
    dmax = np.abs(diff).max()

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    rows = [("Shembulli", before, "terrain", vmin, vmax),
            ("E evoluar", after, "terrain", vmin, vmax),
            ("Ndryshimi", diff, "RdBu_r", -dmax, dmax)]
    for r, (title, data, cmap, lo, hi) in enumerate(rows):
        for m in range(4):
            ax = axes[r, m]
            im = ax.imshow(data[m], cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Forma {m + 1}")
            if m == 0:
                ax.set_ylabel(title)
        fig.colorbar(im, ax=axes[r, :], shrink=0.8)
    name = os.path.basename(run_dir)
    rel = np.abs(diff).mean() / (vmax - vmin) * 100
    fig.suptitle(f"{name}: ndryshimi mesatar = {np.abs(diff).mean():.3f} "
                 f"({rel:.1f}% e shtrirjes se vlerave)")
    out = os.path.join(FIGURES, f"mask_change_{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"U ruajt: {out}  (ndryshimi mesatar {np.abs(diff).mean():.3f}, {rel:.1f}%)")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    plot_convergence()
    for run_dir in sorted(glob.glob(os.path.join(RESULTS, "*_s*"))):
        plot_masks(run_dir)
        plot_mask_change(run_dir)


if __name__ == "__main__":
    main()