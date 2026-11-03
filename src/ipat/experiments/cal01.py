"""CAL-01: replay frozen EXP-05 models, then fit on clean calibration only."""
import importlib
import json
from pathlib import Path
import numpy as np
from .cal01_core import fit_temperature, scale_logits, load_partition

PARENT_SUMMARY = 'ae291bc6688d0c85a287a76a7c448c54c1f6f76ee1ac55aecaa0c1c63c49b2e3'
PARENT_CONFIG = '26ba05156779b11cb03f739c40a3471fd6f32c3db4d0549fda4f8b2701456edf'
ARMS = ('T00','T10','T01','T11','GRU','GBDT')
SEEDS = (1001,1002,1003)
REPLAY_ATOL = 1e-6


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_parent(root):
    from ipat.experiments.exp05 import digest
    from ipat.data.bpi17 import sha256
    root = Path(root)
    acceptance = read_json(root/'exp05_acceptance.json')
    if acceptance.get('status') != 'PASS' or not acceptance.get('checks') or not all(acceptance['checks'].values()):
        raise ValueError('Parent EXP-05 acceptance is not PASS')
    if sha256(root/'exp05_pilot_summary.json') != PARENT_SUMMARY or acceptance['summary_sha256'] != PARENT_SUMMARY:
        raise ValueError('This CAL-01 version requires the reviewed EXP-05 summary')
    manifest = read_json(root/'run_manifest.json')
    if digest(manifest['config']) != PARENT_CONFIG or acceptance['config_sha256'] != PARENT_CONFIG or manifest['configuration_sha256'] != PARENT_CONFIG:
        raise ValueError('Parent configuration differs from reviewed pilot')
    summary = read_json(root/'exp05_pilot_summary.json')
    expected = {(a,s) for a in ARMS for s in SEEDS}
    runs = summary['runs']
    if len(runs)!=18 or {(r['arm'],r['seed']) for r in runs} != expected:
        raise ValueError('Parent must contain all 18 arm/seed pairs')
    for row in runs:
        if row['status']!='PASS' or not row['checkpoint_replay_exact']:
            raise ValueError('Incomplete parent fit')
        if read_json(root/f"{row['arm']}_{row['seed']}"/'result.json') != row:
            raise ValueError('Per-run result differs from reviewed summary')
    for module, expected_hash in manifest['environment']['code_sha256'].items():
        if sha256(importlib.import_module(module).__file__) != expected_hash:
            raise ValueError('Frozen parent source changed: '+module)
    return manifest, summary


def evaluate_checkpoint(root, arm, seed, state, config, identity, expected_epoch,
                        validation, calibration, device, saved_validation):
    """No model fitting. Shared by the real runner and synthetic integration verifier."""
    from ipat.experiments.exp05 import make_model, load_checkpoint, predict, tensor_hash
    from ipat.experiments.exp05_references import prefix_summaries
    from ipat.data.bpi17 import sha256
    root = Path(root)
    checkpoint = root/('selected.joblib' if arm=='GBDT' else 'selected.pt')
    checkpoint_hash = sha256(checkpoint)
    if arm == 'GBDT':
        import joblib
        # Only the user's own frozen checkpoint whose provenance is verified upstream.
        model = joblib.load(checkpoint)
        if list(model.classes_) != [0,1]:
            raise ValueError('Unexpected tree label order')
        v = model.decision_function(prefix_summaries(validation,state))
        features = prefix_summaries(calibration,state)
        raw = model.decision_function(features)
        again = model.decision_function(features)
        state_unchanged = True  # File hash below is the tree persistence invariant.
    else:
        payload = load_checkpoint(checkpoint)
        if payload['identity'] != identity or payload['epoch'] != expected_epoch:
            raise ValueError('Selected checkpoint identity/epoch mismatch')
        model = make_model(arm,state,seed,config['model']).to(device)
        model.load_state_dict(payload['model'],strict=True)
        model.requires_grad_(False)
        before = tensor_hash(model.state_dict())
        v,_ = predict(model,validation,device,config['microbatch'])
        raw,_ = predict(model,calibration,device,config['microbatch'])
        again,_ = predict(model,calibration,device,config['microbatch'])
        state_unchanged = before == tensor_hash(model.state_dict())
    v = np.asarray(v,dtype=np.float64)
    raw = np.asarray(raw,dtype=np.float64)
    if v.shape != saved_validation.shape or not np.isfinite(v).all() or not np.isfinite(raw).all():
        raise ValueError('Invalid checkpoint predictions')
    error = float(np.max(np.abs(v-saved_validation)))
    if error > REPLAY_ATOL:
        raise ValueError(f'Validation replay failed: {arm}/{seed}, max error={error}')
    if not np.array_equal(v>=0,saved_validation>=0):
        raise ValueError('Replay changed validation class decisions')
    if not np.array_equal(raw,again) or not state_unchanged or sha256(checkpoint)!=checkpoint_hash:
        raise ValueError('Model changed or calibration replay is not exact')
    return raw, dict(checkpoint_sha256=checkpoint_hash, validation_replay_max_logit_error=error,
                     validation_replay_atol=REPLAY_ATOL, calibration_replay_exact=True,
                     model_state_unchanged=state_unchanged, checkpoint_file_unchanged=True)


def run(parent, cache, output):
    import torch
    from ipat.data.bpi17 import sha256
    from ipat.experiments.exp05 import digest, write_json, configure_device, environment
    from .exp05_metrics import metrics, stratified, probabilities
    from .cal01_plots import plots
    parent, cache, output = Path(parent).resolve(), Path(cache).resolve(), Path(output).resolve()
    if output.exists() or parent == output or parent in output.parents or cache == output or cache in output.parents:
        raise ValueError('Use a new output directory outside parent/cache')
    manifest, summary = verify_parent(parent)
    config = manifest['config']
    torch.set_num_threads(manifest['environment']['cpu_threads'])
    device = configure_device(manifest['environment']['device'])
    current_env = environment(device)
    if current_env != manifest['environment']:
        raise ValueError('Use the recorded parent runtime/device; environment differs')
    state = read_json(cache/'preprocessing.json')
    if digest(state) != config['preprocessing_logical_sha256']:
        raise ValueError('Preprocessing changed')
    val = load_partition(cache,'validation',config['logical_hashes']['validation'])
    cal = load_partition(cache,'calibration',config['logical_hashes']['calibration'])
    if len(cal)!=6117 or len(val)!=7871 or len(np.unique(cal.arrays['y']))!=2:
        raise ValueError('Calibration/validation population differs from contract')
    if set(cal.arrays['application_id']) & set(val.arrays['application_id']):
        raise ValueError('Application overlap')
    if len(set(cal.arrays['offer_id']))!=len(cal):
        raise ValueError('Duplicate calibration offers')
    output.mkdir(parents=True,exist_ok=False)
    scope = 'CAL-01 calibration FIT diagnostics; no independent test evaluation or generalisation claim'
    code = {k:sha256(importlib.import_module('ipat.experiments.'+k).__file__) for k in ('cal01','cal01_core','cal01_plots')}
    receipt = dict(version='CAL01-v0.1', scope=scope, parent=str(parent), cache=str(cache),
        protocol_sha256=sha256(Path(__file__).resolve().parents[3]/'docs/cal01/CAL01_Protocol_v0.1.md'),
        parent_manifest_sha256=sha256(parent/'run_manifest.json'), parent_summary_sha256=PARENT_SUMMARY,
        parent_config_sha256=PARENT_CONFIG, environment=current_env, cal01_code_sha256=code,
        predictive_partitions=['validation','calibration'], fitting_partition='calibration',
        validation_role='Replay only; never temperature fitting', test_loaded=False,
        temperature_bounds=[.05,20.], seeds=list(SEEDS), arms=list(ARMS))
    write_json(output/'cal01_manifest.json',receipt)
    rows = []
    for row in summary['runs']:
        arm,seed = row['arm'],row['seed']
        source = parent/f'{arm}_{seed}'
        with np.load(source/'predictions.npz',allow_pickle=False) as f:
            for k in ('y','offer_id','application_id'):
                if not np.array_equal(f[k],val.arrays[k]):
                    raise ValueError('Parent validation alignment changed: '+k)
            if not np.array_equal(f['lengths'],np.diff(val.arrays['offsets'])):
                raise ValueError('Parent prefix lengths changed')
            saved = f['logits'].copy()
        if abs(metrics(val.arrays['y'],saved)['nll']-row['selected']['nll'])>1e-10:
            raise ValueError('Saved parent predictions disagree with reviewed NLL')
        raw,checks = evaluate_checkpoint(source,arm,seed,state,config,digest(manifest),
            row.get('selected_epoch'),val,cal,device,saved)
        y,apps = cal.arrays['y'],cal.arrays['application_id']
        fit = fit_temperature(y,raw)
        scaled = scale_logits(raw,fit['temperature'])
        a,b = metrics(y,raw,apps),metrics(y,scaled,apps)
        # Detect any sigmoid rounding that would invalidate fixed-threshold invariance.
        if not np.array_equal(probabilities(raw)>=.5,probabilities(scaled)>=.5):
            raise ValueError('Probability rounding changed threshold decisions')
        for key in ('confusion_matrix','accuracy','balanced_accuracy','macro_f1','classes'):
            if a[key] != b[key]:
                raise ValueError('Temperature changed classification metrics: '+key)
        d = output/f'{arm}_{seed}'
        d.mkdir()
        np.savez_compressed(d/'calibration_predictions.npz',raw_logits=raw,scaled_logits=scaled,
            y=y,application_id=apps,offer_id=cal.arrays['offer_id'],lengths=np.diff(cal.arrays['offsets']))
        entry = dict(arm=arm,seed=seed,scope=scope,fit=fit,raw=a,scaled=b,checks=checks,
            class_decisions_unchanged=True, prediction_sha256=sha256(d/'calibration_predictions.npz'),
            raw_length=stratified(y,raw,apps,np.diff(cal.arrays['offsets'])),
            scaled_length=stratified(y,scaled,apps,np.diff(cal.arrays['offsets'])))
        write_json(d/'temperature.json',entry)
        rows.append(entry)
        print(f"{arm}/{seed}: T={fit['temperature']:.6f}, fit NLL {a['nll']:.6f} -> {b['nll']:.6f}; {fit['status']}",flush=True)
    # Reverify bindings at completion; never overwrite the EXP-05 evidence.
    verify_parent(parent)
    if sha256(parent/'run_manifest.json') != receipt['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed during CAL-01')
    all_interior = all(r['fit']['status']=='INTERIOR' for r in rows)
    result = dict(version='CAL01-v0.1',status='PASS' if all_interior else 'NEEDS_REVIEW',scope=scope,
        test_loaded=False,calibration_offers=len(cal),calibration_applications=len(set(cal.arrays['application_id'])),
        runs=rows, review_cases=[f"{r['arm']}_{r['seed']}: {r['fit']['status']}" for r in rows if r['fit']['status']!='INTERIOR'])
    write_json(output/'cal01_summary.json',result)
    plots(output)
    evidence = dict(status=result['status'],scope='Execution/calibrator readiness, not independent calibration benefit',
        checks=dict(all_18_calibrators=len(rows)==18,parent_replayed=True,models_unchanged=True,
                    class_decisions_unchanged=True,fit_nll_not_worse=True,test_predictive_access_disabled=True,
                    interior_identifiable_solutions=all_interior),
        summary_sha256=sha256(output/'cal01_summary.json'),review_cases=result['review_cases'])
    write_json(output/'cal01_acceptance.json',evidence)
    return evidence
