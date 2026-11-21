"""
run_masks.py - kerkimi i maskave me CMA-ES (484 vlera).

Rrjeti neural, biaset dhe 7 numrat e plote merren nga rover-i shembull i ESA-s dhe mbeten FIKS.
Optimizohen vetem 4 maskat 11x11.

--scenarios subset : kerkimi vlereson ne 6 skenare (5x me shpejt), por 6 skenare mund te
                     mashtrojne (overfitting), prandaj me i miri i çdo iterimi kontrollohet
                     edhe ne 30 dhe ruhet rover-i me i mire ne 30.
--scenarios all    : kerkimi vlereson direkt ne te 30 skenaret (me ngadale, por pa overfitting).

Ruan gjendjen pas çdo iterimi, keshtu qe nje run i nderprere vazhdon aty ku mbeti
(thjesht ekzekutoje perseri me te njejtat argumente).

Shembuj (nga folderi kryesor i projektit):
    python experiments/run_masks.py --init example --seed 1 --max-evals 300 --out results/masks_example_s1
    python experiments/run_masks.py --init example --scenarios all --seed 1 --max-evals 300 --out /content/drive/MyDrive/rovers/masks_all_s1
"""
import os
import sys
import csv
import json
import time
import pickle
import argparse

import numpy as np
import cma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import chromosome as ch
from src.evaluate import evaluate, udp, SUBSET, ALL_SCENARIOS, count_reached


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["example", "random"], default="example",
                   help="maskat fillestare: ato te rover-it shembull ose te rastesishme")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=None,
                   help="hapi fillestar; nese mungon = 0.2 * devijimi standard i maskave fillestare")
    p.add_argument("--popsize", type=int, default=None, help="nese mungon, vlera e paracaktuar e CMA-ES (22)")
    p.add_argument("--scenarios", choices=["subset", "all"], default="subset",
                   help="ku vleresohet gjate kerkimit: 6 skenare ose te 30")
    p.add_argument("--max-evals", type=int, default=300)
    p.add_argument("--out", required=True, help="folderi ku ruhen rezultatet (p.sh. ne Google Drive)")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "checkpoint.pkl")
    log_path = os.path.join(a.out, "log.csv")
    rng = np.random.default_rng(a.seed)

    base = np.asarray(udp.example(), dtype=float)
    example_masks = ch.get_masks(base)
    lb, ub = ch.bounds()
    lb_m, ub_m = lb[ch.MASKS], ub[ch.MASKS]
    search_scenarios = SUBSET if a.scenarios == "subset" else ALL_SCENARIOS

    def objective(m):
        return evaluate(ch.set_masks(base, m), search_scenarios)

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        es, n_evals, best_f, best_m = state["es"], state["n_evals"], state["best_f"], state["best_m"]
        best_full, best_full_m = state["best_full"], state["best_full_m"]
        print(f"Vazhdoj nga checkpoint-i: {n_evals} vleresime, me i miri ne 30 skenare {best_full:.4f}")
    else:
        if a.init == "example":
            m0 = example_masks.copy()
        else:
            m0 = rng.uniform(example_masks.min(), example_masks.max(), ch.NUM_MASK_PARAMS)
        m0 = np.clip(m0, lb_m, ub_m)
        sigma0 = a.sigma if a.sigma is not None else 0.2 * float(np.std(example_masks))
        opts = {"bounds": [list(lb_m), list(ub_m)], "seed": a.seed, "verbose": -9}
        if a.popsize:
            opts["popsize"] = a.popsize
        es = cma.CMAEvolutionStrategy(m0, sigma0, opts)
        n_evals = 0
        best_m = m0
        best_f = objective(m0)         
        best_full_m = m0
        best_full = best_f if a.scenarios == "all" else evaluate(ch.set_masks(base, m0), ALL_SCENARIOS)
        n_evals += 1
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iteration", "n_evals", "best_in_iter_search", "mean_in_iter_search", "best_so_far_search",
                                    "iter_best_on_30", "best_so_far_30", "sigma", "seconds"])
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump({**vars(a), "sigma0": sigma0, "popsize": es.popsize,
                       "start_fitness_search": best_f, "start_fitness_30": best_full}, f, indent=2)
        print(f"Fillimi ({a.init}, {a.scenarios}): kerkimi = {best_f:.4f}, 30 skenare = {best_full:.4f}, "
              f"sigma0 = {sigma0:.4f}, popsize = {es.popsize}")

    while n_evals < a.max_evals and not es.stop():
        t = time.time()
        sols = es.ask()
        fits = [objective(m) for m in sols]
        es.tell(sols, fits)
        n_evals += len(sols)

        i = int(np.argmin(fits))
        if fits[i] < best_f:
            best_f, best_m = fits[i], np.array(sols[i])

        iter_full = fits[i] if a.scenarios == "all" else evaluate(ch.set_masks(base, sols[i]), ALL_SCENARIOS)
        if iter_full < best_full:
            best_full, best_full_m = iter_full, np.array(sols[i])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([es.countiter, n_evals, min(fits), float(np.mean(fits)), best_f,
                                    iter_full, best_full, es.sigma, round(time.time() - t, 1)])
        with open(ckpt_path, "wb") as f:
            pickle.dump({"es": es, "n_evals": n_evals, "best_f": best_f, "best_m": best_m,
                         "best_full": best_full, "best_full_m": best_full_m}, f)
        np.save(os.path.join(a.out, "best_rover.npy"), ch.repair(ch.set_masks(base, best_full_m)))

        print(f"iter {es.countiter:3d} | vleresime {n_evals:4d} | kerkimi: {min(fits):.4f} "
              f"| ne 30: {iter_full:.4f} | me i miri ne 30: {best_full:.4f} | {time.time() - t:.0f} s")

    best_x = ch.repair(ch.set_masks(base, best_full_m))
    full, details = evaluate(best_x, ALL_SCENARIOS, details=True)
    summary = {"best_fitness_search": best_f, "full_fitness_30": full,
               "reached_30": count_reached(details), "n_evals": n_evals, "scenarios": details}
    with open(os.path.join(a.out, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPerfundoi. Me i miri ne kerkim: {best_f:.4f} | rover-i i ruajtur, 30 skenare: {full:.4f} "
          f"| mostra te arritura: {summary['reached_30']}/30")
    print("(Rover-i shembull: 4.563 ne 6 skenare, 7.979 ne 30, 0/30 te arritura)")


if __name__ == "__main__":
    main()