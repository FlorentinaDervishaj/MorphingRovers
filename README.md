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
