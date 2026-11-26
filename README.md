# Morphing Rovers – optimizimi hap pas hapi me CMA-ES

Projekt kursi i bazuar në sfidën **SpOC 2: Morphing Rovers** të ESA-s.

Një rover që ndryshon formën duhet të arrijë mostrën në 30 skenarë në Mars.
Rover-i përshkruhet nga 19 133 numra (4 maska, rrjeti neural, 7 numra të plotë)
dhe optimizohet hap pas hapi me CMA-ES, duke filluar nga rover-i shembull i ESA-s.

## Instalimi
```
py -m pip install -r requirements.txt pytest
py -m pytest -q
```

## Burimet
Simulatori, hartat dhe rover-i shembull janë të ESA Advanced Concepts Team
(https://github.com/esa/SpOC2). Shih `ATTRIBUTION.md`.

## Struktura
```
src/            chromosome.py (ndarja e rover-it), evaluate.py (vlerësimi), parallel.py (paralelizmi)
experiments/    run_masks.py (kërkimi me CMA-ES), validate.py (krahasimi i rover-ave),
                timing.py (kontrolli i skedarëve), baseline.py (analiza e rover-it shembull)
analysis/       plots.py (konvergjenca, maskat), trajectories.py (trajektoret)
tests/          testet, përfshirë krahasimin me kodin e ESA-s
results/        rezultatet e çdo run-i (config.json, log.csv, final.json)
figures/        figurat për raportin
```
## Ekzekutimi
```
py experiments/timing.py         # kontrollon skedarët dhe mat një vlerësim
py experiments/baseline.py       # analiza e rover-it shembull për çdo skenar

# Faza 1: vetëm maskat
py experiments/run_masks.py --scenarios all --seed 1 --workers 5 --out results/masks_all_s1

# Faza 2: maskat + shtresa e daljes, duke filluar nga rover-i më i mirë i fazës 1
py experiments/run_masks.py --free masks+output --start results/masks_all_s2/best_rover.npy --scenarios all --seed 1 --workers 5 --out results/output_s1

# Krahasimi dhe figurat
py experiments/validate.py results/output_s2/best_rover.npy
py analysis/plots.py
py analysis/trajectories.py results/output_s2/best_rover.npy
```
