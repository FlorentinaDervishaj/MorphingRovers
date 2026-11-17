
import json
import numpy as np
import evaluation_code as ec
from evaluation_code import udp

x = udp.example()
score, rec = udp.pretty(x, verbose=True)   # tabela zyrtare e ESA-s, 6 harta x 5 skenare

rows = []
print("\nHarta Skenari  Fitness  Arriti?  Lev. (px)  Dist. fund. / fill.")
for m in range(ec.MAPS_PER_EVALUATION):
    for s in range(ec.SCENARIOS_PER_MAP):
        r = rec[m][s]
        start = np.array([r["x"][0], r["y"][0]])
        end = np.array([r["x"][-1], r["y"][-1]])
        target = ec.SCENARIO_POSITIONS[m][s][2:4].numpy()
        d0 = np.linalg.norm(target - start)
        d = np.linalg.norm(target - end)
        reached = bool(d <= ec.SAMPLE_RADIUS)
        moved = float(np.linalg.norm(end - start))
        fit = float(r["fitness"][0])
        rows.append(dict(map=m + 1, scenario=s + 1, fitness=fit, reached=reached,
                         moved_px=moved, final_dist_ratio=float(d / d0), steps=len(r["x"]) - 1))
        print(f"{m+1:5d} {s+1:8d} {fit:8.3f}  {'PO' if reached else 'jo':7s} {moved:9.1f}  {d/d0:10.2f}")

n_reached = sum(r["reached"] for r in rows)
print(f"\nMesatarja: {score[0]:.3f}   Mostra te arritura: {n_reached}/30")
print("(Rover qe s'levizet: ~10.01. Me e ulet = me mire.)")

with open("baseline_example.json", "w") as f:
    json.dump({"mean_fitness": score[0], "reached": n_reached, "scenarios": rows}, f, indent=2)
print("U ruajt: baseline_example.json")