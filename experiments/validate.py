"""
validate.py - vlereson nje ose me shume rover-a (.npy) ne te 30 skenaret, ne kete kompjuter.

Perdoret qe rezultatet nga kompjutere te ndryshem (Colab, laptop) te krahasohen
ne te njejtin kompjuter. Rover-i shembull i ESA-s vleresohet gjithmone si reference.

Shembull:
    py experiments/validate.py results/masks_all_s1/best_rover.npy results/masks_all_s2/best_rover.npy
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate, udp, ALL_SCENARIOS, count_reached


def main(paths):
    rows = []
    example, det = evaluate(np.asarray(udp.example(), dtype=float), ALL_SCENARIOS, details=True)
    rows.append(("rover-i shembull (ESA)", example, count_reached(det)))
    for p in paths:
        score, det = evaluate(np.load(p), ALL_SCENARIOS, details=True)
        rows.append((p, score, count_reached(det)))

    print(f"\n{'Rover-i':55s} {'Fitness':>8s} {'Ndryshimi':>10s} {'Arritura':>9s}")
    for name, score, reached in rows:
        change = (score - example) / example * 100
        print(f"{name:55s} {score:8.4f} {change:+9.1f}% {reached:6d}/30")

    with open("validation.json", "w") as f:
        json.dump([{"rover": n, "fitness_30": s, "reached": r} for n, s, r in rows], f, indent=2)
    print("\nU ruajt: validation.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
