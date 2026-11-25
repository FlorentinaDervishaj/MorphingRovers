"""
run_masks.py - kerkimi me CMA-ES i maskave (dhe, sipas deshires, i shtreses se daljes se rrjetit).

--free masks         : optimizohen vetem 4 maskat 11x11 (484 vlera); pjesa tjeter mbetet FIKS.
--free masks+output  : maskat + shtresa e daljes se rrjetit (484 + 82 = 566 vlera).
--free masks+output+hidden : + shtresa e fundit e fshehur dhe kujtesa (566 + 3240 = 3806 vlera).
--diagonal           : CMA-ES me kovariance diagonale (sep-CMA); me i shpejte ne shume dimensione.
--start rover.npy    : rover-i nga fillon kerkimi (p.sh. me i miri nga faza e maskave).
                       Nese mungon, fillon nga rover-i shembull i ESA-s.

--scenarios subset : kerkimi vlereson ne 6 skenare (5x me shpejt), por 6 skenare mund te
                     mashtrojne (overfitting), prandaj me i miri i çdo iterimi kontrollohet
                     edhe ne 30 dhe ruhet rover-i me i mire ne 30.
--scenarios all    : kerkimi vlereson direkt ne te 30 skenaret (me ngadale, por pa overfitting).

--workers N        : vlereson N rover-a njekohesisht (p.sh. 5 ne laptop me 6 berthama, 2 ne Colab).

Ruan gjendjen pas çdo iterimi, keshtu qe nje run i nderprere vazhdon aty ku mbeti
(thjesht ekzekutoje perseri me te njejtat argumente).

Shembuj (nga folderi kryesor i projektit):
    py experiments/run_masks.py --scenarios all --seed 2 --workers 5 --out results/masks_all_s2
    py experiments/run_masks.py --free masks+output --start results/masks_all_s2/best_rover.npy
                                --scenarios all --seed 1 --workers 5 --out results/output_s1
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
from src.parallel import ParallelEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--free", choices=["masks", "masks+output", "masks+output+hidden"], default="masks",
                   help="cilat pjese te rover-it optimizohen")
    p.add_argument("--diagonal", action="store_true",
                   help="sep-CMA: vetem variancat, pa korrelacione (per shume dimensione)")
    p.add_argument("--start", default=None, help="rover-i fillestar (.npy); nese mungon, ai shembull i ESA-s")
    p.add_argument("--init", choices=["example", "random"], default="example",
                   help="maskat fillestare: ato te rover-it fillestar ose te rastesishme")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=1.0,
                   help="shumezues i hapit; hapi i çdo pjese = sigma * 0.2 * devijimi standard i asaj pjese")
    p.add_argument("--popsize", type=int, default=None, help="nese mungon, vlera e paracaktuar e CMA-ES")
    p.add_argument("--scenarios", choices=["subset", "all"], default="subset",
                   help="ku vleresohet gjate kerkimit: 6 skenare ose te 30")
    p.add_argument("--max-evals", type=int, default=300)
    p.add_argument("--workers", type=int, default=1, help="sa rover-a vleresohen njekohesisht")
    p.add_argument("--out", required=True, help="folderi ku ruhen rezultatet")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "checkpoint.pkl")
    log_path = os.path.join(a.out, "log.csv")
    rng = np.random.default_rng(a.seed)

    base = np.asarray(np.load(a.start) if a.start else udp.example(), dtype=float)
    idx = ch.free_indices(a.free)                 # pozicionet qe optimizohen
    lb, ub = ch.bounds()
    lb_f, ub_f = lb[idx], ub[idx]

    def build(v):
        """Rover-i i plote: base, me vlerat v ne pozicionet idx."""
        y = base.copy()
        y[idx] = v
        return y

    search_scenarios = SUBSET if a.scenarios == "subset" else ALL_SCENARIOS

    def objective(v):
        return evaluate(build(v), search_scenarios)

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        es, n_evals, best_f, best_v = state["es"], state["n_evals"], state["best_f"], state["best_v"]
        best_full, best_full_v = state["best_full"], state["best_full_v"]
        print(f"Vazhdoj nga checkpoint-i: {n_evals} vleresime, me i miri ne 30 skenare {best_full:.4f}")
    else:
        v0 = base[idx].copy()
        if a.init == "random":
            masks = base[ch.MASK_IDX]
            v0[:ch.NUM_MASK_PARAMS] = rng.uniform(masks.min(), masks.max(), ch.NUM_MASK_PARAMS)
        v0 = np.clip(v0, lb_f, ub_f)

        stds = np.full(len(idx), 0.2 * float(np.std(base[ch.MASK_IDX])))
        n_m, n_o = ch.NUM_MASK_PARAMS, ch.OUTPUT_LAYER_IDX.size
        if a.free in ("masks+output", "masks+output+hidden"):
            stds[n_m:n_m + n_o] = 0.2 * float(np.std(base[ch.OUTPUT_LAYER_IDX]))
        if a.free == "masks+output+hidden":
            stds[n_m + n_o:] = 0.2 * float(np.std(base[ch.HIDDEN_LAYER_IDX]))
        opts = {"bounds": [list(lb_f), list(ub_f)], "seed": a.seed, "verbose": -9, "CMA_stds": list(stds)}
        if a.diagonal:
            opts["CMA_diagonal"] = True
        if a.popsize:
            opts["popsize"] = a.popsize
        es = cma.CMAEvolutionStrategy(v0, a.sigma, opts)

        n_evals = 0
        best_v = v0
        best_f = objective(v0)
        best_full_v = v0
        best_full = best_f if a.scenarios == "all" else evaluate(build(v0), ALL_SCENARIOS)
        n_evals += 1
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iteration", "n_evals", "best_in_iter_search", "mean_in_iter_search",
                                    "best_so_far_search", "iter_best_on_30", "best_so_far_30", "sigma", "seconds"])
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump({**vars(a), "n_free": int(len(idx)), "popsize": es.popsize,
                       "step_masks": float(stds[0]), "step_last_block": float(stds[-1]),
                       "start_fitness_search": best_f, "start_fitness_30": best_full}, f, indent=2)
        print(f"Fillimi ({a.free}, {a.scenarios}, {len(idx)} vlera): kerkimi = {best_f:.4f}, "
              f"30 skenare = {best_full:.4f}, popsize = {es.popsize}")

    pe = ParallelEvaluator(a.workers) if a.workers > 1 else None
    if pe:
        print(f"Vleresim paralel me {a.workers} procese")
    while n_evals < a.max_evals and not es.stop():
        t = time.time()
        sols = es.ask()
        if pe:
            fits = pe.map([build(v) for v in sols], search_scenarios)
        else:
            fits = [objective(v) for v in sols]
        es.tell(sols, fits)
        n_evals += len(sols)

        i = int(np.argmin(fits))
        if fits[i] < best_f:
            best_f, best_v = fits[i], np.array(sols[i])

        iter_full = fits[i] if a.scenarios == "all" else evaluate(build(sols[i]), ALL_SCENARIOS)
        if iter_full < best_full:
            best_full, best_full_v = iter_full, np.array(sols[i])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([es.countiter, n_evals, min(fits), float(np.mean(fits)), best_f,
                                    iter_full, best_full, es.sigma, round(time.time() - t, 1)])
        with open(ckpt_path, "wb") as f:
            pickle.dump({"es": es, "n_evals": n_evals, "best_f": best_f, "best_v": best_v,
                         "best_full": best_full, "best_full_v": best_full_v}, f)
        np.save(os.path.join(a.out, "best_rover.npy"), ch.repair(build(best_full_v)))

        print(f"iter {es.countiter:3d} | vleresime {n_evals:4d} | kerkimi: {min(fits):.4f} "
              f"| ne 30: {iter_full:.4f} | me i miri ne 30: {best_full:.4f} | {time.time() - t:.0f} s")

    if pe:
        pe.close()

    best_x = ch.repair(build(best_full_v))
    full, details = evaluate(best_x, ALL_SCENARIOS, details=True)
    summary = {"best_fitness_search": best_f, "full_fitness_30": full,
               "reached_30": count_reached(details), "n_evals": n_evals, "scenarios": details}
    with open(os.path.join(a.out, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPerfundoi. Me i miri ne kerkim: {best_f:.4f} | rover-i i ruajtur, 30 skenare: {full:.4f} "
          f"| mostra te arritura: {summary['reached_30']}/30")


if __name__ == "__main__":
    main()