"""Reference depth law for native sparse Circle (no tensor/runtime dependency)."""
from functools import lru_cache
import math


@lru_cache(maxsize=64)
def depth_reference(minimum, maximum, mean):
    """Maximum-entropy law on [minimum, maximum] with the requested mean.

    p(d) is proportional to exp(tilt*d). This is a population prior, never
    a sequence-dependent top-k quota. Return probabilities and P(D >= s).
    """
    if not minimum < mean < maximum:
        raise ValueError('Circle target must be strictly inside its depth range')
    depths = list(range(minimum, maximum+1))

    def law(tilt):
        logits = [tilt*(d-mean) for d in depths]
        anchor = max(logits)
        weights = [math.exp(x-anchor) for x in logits]
        total = sum(weights)
        return [x/total for x in weights]

    lo, hi = -64., 64.
    for _ in range(90):
        mid = (lo+hi)/2
        p = law(mid)
        if sum(d*q for d,q in zip(depths,p)) < mean: lo = mid
        else: hi = mid
    probabilities = law((lo+hi)/2)
    survival = [1. if s <= minimum else sum(probabilities[s-minimum:])
                for s in range(1, maximum+1)]
    return tuple(probabilities), tuple(survival)
