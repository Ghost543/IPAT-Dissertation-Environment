"""Detached binary temperature fitting; NumPy only, no model optimiser."""
import numpy as np

T_MIN, T_MAX = 0.05, 20.0
NLL_TOL = 1e-10


def aligned(y, logits):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(logits, dtype=np.float64)
    if y.ndim != 1 or x.shape != y.shape or not len(y):
        raise ValueError('Nonempty aligned one-dimensional labels/logits required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(x).all():
        raise ValueError('Binary labels and finite logits required')
    return y, x


def scale_logits(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and strictly positive')
    x = np.asarray(logits, dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        scaled = x / temperature
    if not np.isfinite(scaled).all():
        raise ValueError('Scaled logits are not finite')
    # Check the logit decision, including a zero logit assigned to class 1.
    if not np.array_equal(x >= 0, scaled >= 0):
        raise ValueError('Floating-point scaling changed a class decision')
    return scaled


def nll(y, logits):
    y, x = aligned(y, logits)
    # Signed-margin form avoids catastrophic cancellation for correct large logits.
    return float(np.logaddexp(0., -(2*y-1)*x).mean())


def fit_temperature(y, logits):
    y, x = aligned(y, logits)
    if len(np.unique(y)) != 2:
        raise ValueError('CAL-01 requires both labels on its fitting partition')
    if not np.any(x):
        return dict(temperature=1., inverse_temperature=1., status='UNIDENTIFIABLE',
                    boundary=False, iterations=0, raw_nll=nll(y,x), fitted_nll=nll(y,x),
                    gradient_at_solution=0., bounds=[T_MIN,T_MAX])
    # The objective is convex in beta=1/T; its derivative is monotone.
    # Bounded beta search is a numerical engineering restriction, not a theorem.
    def derivative(beta):
        z = scale_logits(x, 1./beta)
        margin = (2*y-1)*z
        return float(np.mean(-(2*y-1)*x * np.exp(-np.logaddexp(0., margin))))
    lo, hi = 1./T_MAX, 1./T_MIN
    iterations = 0
    if derivative(lo) >= 0:
        beta = lo
    elif derivative(hi) <= 0:
        beta = hi
    else:
        for iterations in range(1, 101):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
            if hi-lo <= 1e-12:
                break
        beta = (lo+hi)/2
    # Include the uncalibrated solution explicitly. Never accept worse fit NLL.
    candidates = [1., beta, 1./T_MAX, 1./T_MIN]
    beta = min(candidates, key=lambda b: nll(y,scale_logits(x,1./b)))
    temperature = 1./beta
    raw, fitted = nll(y,x), nll(y,scale_logits(x,temperature))
    if fitted > raw + NLL_TOL:
        raise AssertionError('Calibration objective worsened')
    boundary = temperature in (T_MIN,T_MAX)
    return dict(temperature=float(temperature), inverse_temperature=float(beta),
                status='BOUNDARY' if boundary else 'INTERIOR', boundary=boundary,
                iterations=iterations, raw_nll=raw, fitted_nll=fitted,
                gradient_at_solution=derivative(beta), bounds=[T_MIN,T_MAX])


def load_partition(cache, split, expected_hash):
    # Reject forbidden partitions before importing/loading any dataset.
    if split not in ('validation','calibration'):
        raise ValueError('CAL-01 loads validation for replay and calibration for fitting only')
    from pathlib import Path
    from ipat.data.bpi17 import PackedOffers, logical_hash
    data = PackedOffers.load(Path(cache)/(split+'.npz'))
    if logical_hash(data.arrays) != expected_hash:
        raise ValueError('Logical cache hash mismatch: '+split)
    return data
