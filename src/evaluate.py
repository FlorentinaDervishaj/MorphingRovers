
import numpy as np
import evaluation_code as ec
from evaluation_code import udp
from src.chromosome import repair

ALL_SCENARIOS = [(m, s) for m in range(ec.MAPS_PER_EVALUATION) for s in range(ec.SCENARIOS_PER_MAP)]
SUBSET = [(m, 0) for m in range(ec.MAPS_PER_EVALUATION)] 

PENALTY = 1e9   


def evaluate(x, scenarios=ALL_SCENARIOS, details=False, fix=True):

    if fix:
        x = repair(x)
    try:
        rover = ec.Rover(x)   
        results = []
        for m, s in scenarios:
            d, t = udp.run_single_scenario(rover, m, s)
            d, t = float(d), float(t)
            results.append({"map": m + 1, "scenario": s + 1,
                            "fitness": (1 + d) * t, "reached": d == 0.0})
        score = float(np.mean([r["fitness"] for r in results]))
        if not np.isfinite(score):
            score = PENALTY
    except Exception as e:
        print("Vleresimi deshtoi:", e)
        score, results = PENALTY, []

    return (score, results) if details else score


def count_reached(results):
    return sum(r["reached"] for r in results)