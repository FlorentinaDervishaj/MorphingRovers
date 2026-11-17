import os
import time

DATA = os.path.join(".", "data", "spoc2", "morphing")


needed = [
    os.path.join(DATA, "coordinates.txt"),
    os.path.join(DATA, "example_rover.npy"),
] + [os.path.join(DATA, "Maps", f"Map{i}.jpg") for i in range(1, 7)]

missing = [p for p in needed if not os.path.exists(p)]
if missing:
    print("Missing files:")
    for p in missing:
        print("  ", p)
    raise SystemExit(1)
print("All data files found.")


t0 = time.time()
from evaluation_code import udp
print(f"Loaded maps and simulator in {time.time() - t0:.1f} s")


x = udp.example()
print("Chromosome length:", len(x), "(expected 19133)")

t1 = time.time()
score = udp.fitness(x)[0]
elapsed = time.time() - t1

print(f"\nExample rover mean fitness: {score:.3f}")
print("(A rover that never moves scores about 10.01. Lower is better.)")
print(f"One full evaluation took {elapsed:.1f} seconds")