"""Detached EXP-06 evaluation and shared application-cluster resampling. No fitting."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .exp05_metrics import metrics, stratified, probabilities, losses
from .cal01_core import scale_logits
from .exp05_io import atomic_write

ARMS = ('T00', 'T10', 'T01', 'T11', 'GRU', 'GBDT')
SEEDS = (1001, 1002, 1003)
VARIANTS = ('raw', 'scaled')
PROTOCOL_SHA = 'ea59ef205c0531c00c6739f10548485567e62f7865e10ac7865ef53c4bef486d'
REPLAY_ATOL = 1e-6
META = ('y', 'offer_id', 'application_id', 'lengths')
SCOPE = 'Held-out clean test; fixed EXP-05 checkpoints and CAL-01 temperatures'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    atomic_write(path, lambda f: f.write(data))


def save_npz(path, **arrays):
    atomic_write(path, lambda f: np.savez_compressed(f, **arrays))


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(data):
    a = data.arrays
    result = {k: a[k] for k in META[:3]}
    result['lengths'] = np.diff(a['offsets'])
    validate_metadata(result)
    return result


def validate_metadata(a):
    n = len(a['y'])
    require(n > 0 and all(np.asarray(a[k]).shape == (n,) for k in META), 'Invalid metadata dimensions')
    require(np.isin(a['y'], [0, 1]).all(), 'Nonbinary labels')
    require(np.issubdtype(a['lengths'].dtype, np.integer) and (a['lengths'] > 0).all(), 'Invalid lengths')
    for k in ('offer_id', 'application_id'):
        require(a[k].dtype.kind in 'US' and all(str(x) for x in a[k]), 'Invalid identifier: ' + k)
    require(len(np.unique(a['offer_id'])) == n, 'Duplicate offers')


def aligned(a, b):
    for k in META:
        require(np.array_equal(a[k], b[k]), 'Prediction alignment mismatch: ' + k)


def disjoint(partitions):
    seen_apps, seen_offers = set(), set()
    for name, a in partitions.items():
        validate_metadata(a)
        apps, offers = set(a['application_id']), set(a['offer_id'])
        require(not (seen_apps & apps), 'Application overlap: ' + name)
        require(not (seen_offers & offers), 'Offer overlap: ' + name)
        seen_apps.update(apps)
        seen_offers.update(offers)


def load_partition(cache, split, expected):
    require(split in ('validation', 'calibration', 'test'), 'EXP-06 predictive loader forbids training')
    from ipat.data.bpi17 import PackedOffers, logical_hash
    data = PackedOffers.load(Path(cache) / (split + '.npz'))
    require(logical_hash(data.arrays) == expected, 'Logical cache hash mismatch: ' + split)
    metadata(data)
    return data


def replay(actual, saved, name):
    actual, saved = np.asarray(actual), np.asarray(saved)
    require(actual.ndim == 1 and actual.size > 0 and actual.shape == saved.shape, name + ': replay shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(saved).all(), name + ': nonfinite replay')
    error = float(np.max(np.abs(actual - saved)))
    require(error <= REPLAY_ATOL, name + ': replay exceeds 1e-6')
    require(np.array_equal(actual >= 0, saved >= 0), name + ': logit decisions changed')
    require(np.array_equal(probabilities(actual) >= .5, probabilities(saved) >= .5), name + ': probability decisions changed')
    return error


def validate_temperature(entry, expected_row, checkpoint):
    require(entry == expected_row, 'Temperature record differs from reviewed summary')
    t = entry['fit']['temperature']
    require(np.isfinite(t) and .05 < t < 20 and entry['fit']['status'] == 'INTERIOR', 'Temperature not ready')
    require(entry['checks']['checkpoint_sha256'] == sha(checkpoint), 'Checkpoint/temperature binding mismatch')
    return float(t)


def prediction_record(meta, raw, temperature):
    validate_metadata(meta)
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.shape == meta['y'].shape and np.isfinite(raw).all(), 'Invalid test logits')
    scaled = scale_logits(raw, temperature)
    p, q = probabilities(raw), probabilities(scaled)
    require(np.isfinite(p).all() and np.isfinite(q).all(), 'Nonfinite probabilities')
    require(np.array_equal(raw >= 0, p >= .5) and np.array_equal(scaled >= 0, q >= .5),
            'Sigmoid rounding disagrees with logit decisions')
    require(np.array_equal(p >= .5, q >= .5), 'Scaling changed class decisions')
    return dict(meta, raw_logits=raw, scaled_logits=scaled, raw_probability=p, scaled_probability=q)


def evaluate_record(a):
    validate_metadata(a)
    result = {}
    for variant in VARIANTS:
        x = a[variant + '_logits']
        require(np.isfinite(x).all(), 'Nonfinite saved logits')
        require(np.array_equal(a[variant + '_probability'], probabilities(x)), 'Saved probabilities differ from logits')
        result[variant] = metrics(a['y'], x, a['application_id'])
        result[variant + '_length'] = stratified(a['y'], x, a['application_id'], a['lengths'])
    for k in ('confusion_matrix', 'accuracy', 'balanced_accuracy', 'macro_f1', 'classes'):
        require(result['raw'][k] == result['scaled'][k], 'Scaling changed class metric: ' + k)
    # Probability saturation can introduce ranking ties; record, do not hide them.
    result['ranking_diagnostics'] = {
        'unique_raw_logits': int(np.unique(a['raw_logits']).size),
        'unique_raw_probabilities': int(np.unique(a['raw_probability']).size),
        'unique_scaled_probabilities': int(np.unique(a['scaled_probability']).size),
        'auroc_changed': result['raw']['auroc'] != result['scaled']['auroc'],
        'average_precision_changed': result['raw']['average_precision'] != result['scaled']['average_precision']}
    return result


def cluster_draws(apps, draws=2000, seed=60601):
    unique, inverse = np.unique(apps, return_inverse=True)
    require(len(unique) > 0 and draws > 0, 'Empty bootstrap')
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(unique), size=(draws, len(unique)), dtype=np.int64)
    return unique, inverse, indices


def cluster_interval(delta_by_seed, inverse, indices):
    d = np.asarray(delta_by_seed, dtype=np.float64)
    require(d.ndim == 2 and d.shape[1] == len(inverse) and np.isfinite(d).all(), 'Invalid paired losses')
    require(indices.ndim == 2 and indices.shape[1] == len(np.unique(inverse)), 'Invalid bootstrap dimensions')
    counts = np.bincount(inverse)
    require((indices >= 0).all() and (indices < len(counts)).all(), 'Invalid bootstrap application index')
    totals = np.bincount(inverse, weights=d.mean(axis=0), minlength=len(counts))
    # Bound intermediate memory independently of the real cohort size.
    values = []
    for start in range(0, len(indices), 64):
        ix = indices[start:start+64]
        values.extend((totals[ix].sum(axis=1) / counts[ix].sum(axis=1)).tolist())
    seeds = d.mean(axis=1)
    return dict(mean_delta=float(seeds.mean()), by_seed=seeds.tolist(),
                sample_std=float(seeds.std(ddof=1)) if len(seeds) > 1 else None,
                percentile_95=np.quantile(values, [.025, .975], method='linear').tolist(),
                draws=len(indices), scope='Descriptive application-cluster interval conditional on frozen predictors')


def flatten(value, prefix=''):
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + '.'))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) or item is None:
            out[name] = item
    return out


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda f: f.write(buffer.getvalue().encode('utf-8')))


def aggregate(root, *, fixture_only=False):
    """Recompute all tables from persisted predictions; bootstrap settings are fixed."""
    root = Path(root)
    rows, arrays, lm = [], {}, {}
    ref = None
    for arm in ARMS:
        for seed in SEEDS:
            d = root / f'{arm}_{seed}'
            a = load_npz(d / 'test_predictions.npz')
            r = read_json(d / 'result.json')
            require((r['arm'], r['seed']) == (arm, seed), 'Wrong result identity')
            require(r['prediction_sha256'] == sha(d / 'test_predictions.npz'), 'Saved prediction hash changed')
            aligned(a, ref if ref is not None else a)
            ref = a if ref is None else ref
            calc = evaluate_record(a)
            for k, v in calc.items():
                require(r[k] == v, 'Saved metric disagrees with predictions: ' + k)
            require(np.array_equal(scale_logits(a['raw_logits'], r['temperature']), a['scaled_logits']), 'Saved temperature application differs')
            arrays[arm, seed] = a
            rows.append(r)
            for variant in VARIANTS:
                lm[arm, seed, variant] = losses(a['y'], a[variant + '_logits'])
    q = 13234 / 19071
    dummy_logits = np.full(len(ref['y']), np.log(q) - np.log1p(-q))
    dummy_loss = losses(ref['y'], dummy_logits)
    dummy = metrics(ref['y'], dummy_logits, ref['application_id'])
    dummy['probability_from_train'] = q
    dummy['length_strata'] = stratified(ref['y'], dummy_logits, ref['application_id'], ref['lengths'])
    apps, inv, draws = cluster_draws(ref['application_id'])
    save_npz(root / 'bootstrap_draws.npz', application_id=apps, indices=draws)
    comparisons = {}
    def ci(name, deltas):
        comparisons[name] = cluster_interval(np.stack(deltas), inv, draws)
    for variant in VARIANTS:
        for arm in ARMS:
            if arm != 'T00':
                ci(f'{variant}:{arm}-T00', [lm[arm,s,variant]-lm['T00',s,variant] for s in SEEDS])
            ci(f'{variant}:{arm}-Dummy', [lm[arm,s,variant]-dummy_loss for s in SEEDS])
    for arm in ARMS:
        ci(f'calibration:{arm}:scaled-raw', [lm[arm,s,'scaled']-lm[arm,s,'raw'] for s in SEEDS])
    ci('raw:factorial_interaction', [lm['T11',s,'raw']-lm['T10',s,'raw']-lm['T01',s,'raw']+lm['T00',s,'raw'] for s in SEEDS])
    summary = {}
    for arm in ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        summary[arm] = {}
        for variant in VARIANTS:
            flat = [flatten(r[variant]) for r in rr]
            summary[arm][variant] = {k: {
                'by_seed': [v[k] for v in flat],
                'mean': float(np.mean([v[k] for v in flat])) if all(v[k] is not None for v in flat) else None,
                'sample_std': float(np.std([v[k] for v in flat], ddof=1)) if all(v[k] is not None for v in flat) else None}
                for k in flat[0]}
        summary[arm]['identical_seed_raw_predictions'] = all(np.array_equal(arrays[arm,1001]['raw_logits'],arrays[arm,s]['raw_logits']) for s in SEEDS[1:])
    result = dict(version='EXP06-v0.1', status='EVALUATED_REPORT_PENDING', fixture_only=fixture_only,
        scope='SYNTHETIC FIXTURE ONLY' if fixture_only else SCOPE,
        n=len(ref['y']), applications=len(apps), runs=rows, summary=summary, dummy=dummy,
        primary_contrast='raw:T11-T00', comparisons=comparisons,
        bootstrap=dict(seed=60601, generator='PCG64', draws=2000, file_sha256=sha(root/'bootstrap_draws.npz'),
                       indices_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
        seed_aggregation='Mean of scores; no probability ensemble',
        predictive_partitions=['validation_replay','calibration_replay','test'], fitted_on_test=False)
    write_json(root/'exp06_summary.json', result)
    csv_write(root/'metrics.csv', [dict(arm=r['arm'],seed=r['seed'],variant=v,**flatten(r[v])) for r in rows for v in VARIANTS])
    csv_write(root/'comparisons.csv', [dict(contrast=k,mean_delta=v['mean_delta'],sample_std=v['sample_std'],
        lower=v['percentile_95'][0],upper=v['percentile_95'][1],**{f'seed_{s}':x for s,x in zip(SEEDS,v['by_seed'])}) for k,v in comparisons.items()])
    csv_write(root/'length_metrics.csv', [dict(arm=r['arm'],seed=r['seed'],variant=v,stratum=k,**flatten(m))
        for r in rows for v in VARIANTS for k,m in r[v+'_length'].items()])
    return result
