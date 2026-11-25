"""
trajectories.py - trajektoret e rover-it shembull te ESA-s dhe te nje rover-i tjeter ne te 30 skenaret.

Figura: 6 rreshta (hartat) x 5 kolona (skenaret). Ne çdo panel:
    vija gri       = rover-i shembull i ESA-s
    vija e kuqe    = rover-i im
    rrethi blu     = pika e nisjes
    diamanti i kuq = mostra
Titulli i panelit tregon fitness-in e te dy rover-ave; "✓" do te thote qe rover-i im e arriti mostren.

Perdorimi (nga folderi kryesor; zgjat ~1 min, sepse ekzekuton simulatorin dy here):
    py analysis/trajectories.py results/output_s2/best_rover.npy
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation_code as ec
from evaluation_code import udp

FIGURES = "figures"
MARGIN = 60   


def run(x):
    score, rec = udp.pretty(np.asarray(x, dtype=float), verbose=False)
    return score[0], rec


def main(path):
    os.makedirs(FIGURES, exist_ok=True)
    ex_score, ex = run(udp.example())
    my_score, my = run(np.load(path))

    nm, ns = ec.MAPS_PER_EVALUATION, ec.SCENARIOS_PER_MAP
    fig, axes = plt.subplots(nm, ns, figsize=(15, 18))
    for m in range(nm):
        hmap = udp.env.heightmaps[m].flip(0).numpy()      
        for s in range(ns):
            ax = axes[m, s]
            sx, sy, tx, ty = [float(v) for v in ec.SCENARIO_POSITIONS[m][s]]
            exx, exy = np.array(ex[m][s]["x"]), np.array(ex[m][s]["y"])
            myx, myy = np.array(my[m][s]["x"]), np.array(my[m][s]["y"])

            ax.imshow(hmap, origin="lower", cmap="gray")
            ax.plot(exx, exy, color="0.6", lw=1.2, ls="--")
            ax.plot(myx, myy, color="tab:red", lw=1.5)
            ax.plot(sx, sy, "o", color="tab:blue", ms=6)
            ax.plot(tx, ty, "D", color="red", ms=7, mec="k")

            xs = np.concatenate([exx, myx, [sx, tx]])
            ys = np.concatenate([exy, myy, [sy, ty]])
            ax.set_xlim(xs.min() - MARGIN, xs.max() + MARGIN)
            ax.set_ylim(ys.min() - MARGIN, ys.max() + MARGIN)
            ax.set_xticks([]); ax.set_yticks([])

            f_ex = ex[m][s]["fitness"][0]
            f_my = my[m][s]["fitness"][0]
            reached = np.hypot(myx[-1] - tx, myy[-1] - ty) <= ec.SAMPLE_RADIUS
            ax.set_title(f"H{m + 1} S{s + 1}: {f_ex:.1f} → {f_my:.1f}" + (" ✓" if reached else ""),
                         fontsize=10)

    name = os.path.basename(os.path.dirname(path)) or "rover"
    fig.suptitle(f"Trajektoret: shembulli i ESA-s (gri, {ex_score:.3f}) "
                 f"dhe {name} (e kuqe, {my_score:.3f})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(FIGURES, f"trajectories_{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("U ruajt:", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
