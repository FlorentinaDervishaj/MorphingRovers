"""
parallel.py - vleresimi i disa rover-ave njekohesisht ne disa berthama te procesorit.

Çdo proces (worker) e ngarkon simulatorin e ESA-s nje here ne fillim (~20 s),
pastaj vlereson rover-at qe i dergohen. PyTorch kufizohet ne 1 thread per proces,
qe proceset te mos e pengojne njeri-tjetrin.

Perdorimi:
    from src.parallel import ParallelEvaluator
    with ParallelEvaluator(workers=5) as pe:
        fits = pe.map(lista_e_kromozomeve, scenarios)

Ne Windows, skripti qe e perdor DUHET te kete:  if __name__ == "__main__":
"""
import os
from multiprocessing import get_context

_evaluate = None   


def _init_worker():
    """Ekzekutohet nje here ne çdo worker."""
    global _evaluate
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from src.evaluate import evaluate  
    _evaluate = evaluate


def _work(job):
    x, scenarios = job
    return _evaluate(x, scenarios)


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


class ParallelEvaluator:
    def __init__(self, workers=None):
        self.workers = workers or default_workers()
        self.pool = get_context("spawn").Pool(self.workers, initializer=_init_worker)

    def map(self, xs, scenarios):
        return self.pool.map(_work, [(x, scenarios) for x in xs], chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
