"""
chromosome.py - ndan dhe bashkon kromozomin e rover-it (19 133 numra).

Rendi i kromozomit, sipas kodit te ESA-s (evaluation_code.py):
    [ 4 maska 11x11 (484) | biaset (146) | peshat (18 496) | 7 numra te plote ]

Indekset:
    maskat   0     .. 483
    biaset   484   .. 629
    peshat   630   .. 19125
    numrat   19126 .. 19132   (pooling1, pooling2, aktivizimi1..5)
"""
import numpy as np

# Konstantet (te njejta me ato te evaluation_code.py, kontrollohen ne teste)
NUM_MODES = 4
MASK_SIZE = 11
NUM_MASK_PARAMS = NUM_MODES * MASK_SIZE ** 2      # 484
NUM_BIASES = 146
NUM_WEIGHTS = 18496
NUM_NN_PARAMS = NUM_BIASES + NUM_WEIGHTS          # 18 642
NUM_INTEGERS = 7
TOTAL_LENGTH = NUM_MASK_PARAMS + NUM_NN_PARAMS + NUM_INTEGERS   # 19 133

# Kufijte e ESA-s
FLOAT_MIN, FLOAT_MAX = -100.0, 100.0
CENTRE_MIN = 1e-16
INT_LOWER = np.array([0, 0, 0, 0, 0, 0, 0])
INT_UPPER = np.array([1, 1, 6, 6, 6, 6, 4])

# Pozicionet ku fillon secila pjese
MASKS = slice(0, NUM_MASK_PARAMS)
BIASES = slice(NUM_MASK_PARAMS, NUM_MASK_PARAMS + NUM_BIASES)
WEIGHTS = slice(NUM_MASK_PARAMS + NUM_BIASES, NUM_MASK_PARAMS + NUM_NN_PARAMS)
INTEGERS = slice(NUM_MASK_PARAMS + NUM_NN_PARAMS, TOTAL_LENGTH)

# Indekset e qendres se seciles maske (duhet te jene >= CENTRE_MIN)
MASK_CENTRES = [m * MASK_SIZE ** 2 + MASK_SIZE ** 2 // 2 for m in range(NUM_MODES)]  # 60, 181, 302, 423


def _check_length(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (TOTAL_LENGTH,):
        raise ValueError(f"Kromozomi duhet te kete {TOTAL_LENGTH} vlera, jo {x.shape}")
    return x


def split(x):
    """Ndan kromozomin ne pjeset e tij. Kthen kopje, jo pamje te vektorit origjinal."""
    x = _check_length(x)
    return {
        "masks": x[MASKS].reshape(NUM_MODES, MASK_SIZE, MASK_SIZE).copy(),
        "biases": x[BIASES].copy(),
        "weights": x[WEIGHTS].copy(),
        "integers": x[INTEGERS].copy(),
    }


def merge(masks, biases, weights, integers):
    """Bashkon pjeset prape ne nje kromozom me 19 133 vlera."""
    parts = [np.asarray(masks, dtype=float).ravel(),
             np.asarray(biases, dtype=float).ravel(),
             np.asarray(weights, dtype=float).ravel(),
             np.asarray(integers, dtype=float).ravel()]
    x = np.concatenate(parts)
    return _check_length(x)


def get_masks(x):
    """Kthen vetem maskat si vektor me 484 vlera."""
    return _check_length(x)[MASKS].copy()


def set_masks(x, masks):
    """Kthen kromozom te ri ku maskat jane zevendesuar; pjeset e tjera mbeten te njejta."""
    y = _check_length(x).copy()
    masks = np.asarray(masks, dtype=float).ravel()
    if masks.size != NUM_MASK_PARAMS:
        raise ValueError(f"Maskat duhet te kene {NUM_MASK_PARAMS} vlera, jo {masks.size}")
    y[MASKS] = masks
    return y


def repair(x):
    """
    E ben kromozomin te vlefshem sipas rregullave te ESA-s:
    - te gjitha vlerat ne [-100, 100]
    - qendra e çdo maske >= 1e-16
    - 7 numrat e fundit te rrumbullakosur dhe brenda kufijve te tyre
    """
    y = np.clip(_check_length(x), FLOAT_MIN, FLOAT_MAX)
    for c in MASK_CENTRES:
        y[c] = max(y[c], CENTRE_MIN)
    y[INTEGERS] = np.clip(np.round(y[INTEGERS]), INT_LOWER, INT_UPPER)
    return y


def bounds():
    """Kufijte e poshtem dhe te siperm, njesoj si udp.get_bounds()."""
    lb = np.full(TOTAL_LENGTH, FLOAT_MIN)
    ub = np.full(TOTAL_LENGTH, FLOAT_MAX)
    lb[MASK_CENTRES] = CENTRE_MIN
    lb[INTEGERS] = INT_LOWER
    ub[INTEGERS] = INT_UPPER
    return lb, ub


# --- Shtresa e daljes se rrjetit (output: 40 -> 2) ---
# Ne kodin e ESA-s parametrat renditen sipas shtresave: inp, conv, conv2, lin2, lin3, lin4, recurr, output.
# Prandaj shtresa e daljes eshte e fundit: 2 biaset e fundit dhe 80 peshat e fundit (2 x 40).
OUTPUT_BIAS_IDX = np.arange(BIASES.stop - 2, BIASES.stop)              # 628, 629
OUTPUT_WEIGHT_IDX = np.arange(WEIGHTS.stop - 2 * 40, WEIGHTS.stop)     # 19046 .. 19125
OUTPUT_LAYER_IDX = np.concatenate([OUTPUT_BIAS_IDX, OUTPUT_WEIGHT_IDX])  # 82 vlera
MASK_IDX = np.arange(MASKS.start, MASKS.stop)                           # 484 vlera


def free_indices(name):
    """Indekset qe optimizohen: 'masks' (484) ose 'masks+output' (484 + 82 = 566)."""
    if name == "masks":
        return MASK_IDX
    if name == "masks+output":
        return np.concatenate([MASK_IDX, OUTPUT_LAYER_IDX])
    raise ValueError(f"I panjohur: {name}")
