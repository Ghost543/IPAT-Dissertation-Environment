"""EXP-06 frozen clean evaluation. No training or temperature-fitting entry point."""
import importlib
import time
from pathlib import Path
import numpy as np
from .exp06_core import (ARMS, SEEDS, META, SCOPE, PROTOCOL_SHA, sha, read_json, write_json,
    require, load_npz, save_npz, metadata, aligned, disjoint, load_partition, replay,
    validate_temperature, prediction_record, evaluate_record, aggregate)

CAL_CODE = {
    'cal01.py':'34c80d16cf66aefe383f8cabb19e1b0c675f1a08489bc29be5c02f6d80aaccda',
    'cal01_core.py':'4740fe763eb0a0eb8f859f92e95618741513ba9d73efb66d722626b59645d9a0',
    'cal01_plots.py':'02074300b56f6c038f8303478bf165928752e42797e1ded6598628a4edba8459'}


def project_root():
    return Path(__file__).resolve().parents[3]


def code_hashes():
    root = project_root()
    names = ['src/ipat/experiments/'+n+'.py' for n in ('exp06','exp06_core','exp06_plots','exp06_config_binding')]
    names += ['scripts/run_exp06.py','scripts/verify_exp06.py','tests/test_exp06.py']
    names += ['docs/exp06/EXP06_Clean_Test_Protocol_v0.1.md','configs/exp06_clean_test_protocol_v0.1.json']
    return {n:sha(root/n) for n in names}


def verify_local_receipt(path):
    receipt = read_json(path)
    require(receipt.get('status') == 'PASS' and receipt.get('fixture_only') is True, 'Run the EXP-06 synthetic verifier first')
    needed = ('unit_tests','all_18_checkpoint_paths','parent_replays','exact_test_replay',
              'immutability','negative_guards','saved_metric_recompute','figure_exports','offline_html')
    require(all(receipt.get('checks',{}).get(k) is True for k in needed), 'Incomplete synthetic verification')
    require(receipt.get('code_sha256') == code_hashes(), 'EXP-06 code changed after local verification')


def preflight(parent, calibration, cache):
    """Verify real parent lineage and all 18 replays before test predictive loading."""
    import torch
    from .cal01 import verify_parent, evaluate_checkpoint
    from .exp05 import digest, environment, configure_device
    from .exp05_metrics import metrics
    from ipat.data.bpi17 import PackedOffers, logical_hash
    root = project_root()
    protocol = root/'docs/exp06/EXP06_Clean_Test_Protocol_v0.1.md'
    require(sha(protocol) == PROTOCOL_SHA, 'Approved protocol changed')
    spec = read_json(root/'configs/exp06_clean_test_protocol_v0.1.json')
    # Pin the complete approved configuration as well as the document; no CLI metric overrides.
    from .exp06_config_binding import CONFIG_SHA
    require(sha(root/'configs/exp06_clean_test_protocol_v0.1.json') == CONFIG_SHA, 'Approved configuration changed')
    for name, expected in spec['evidence_sha256'].items():
        source = calibration if name.startswith('cal01') else parent
        require(sha(source/name) == expected, 'Reviewed evidence changed: '+name)
    manifest, summary = verify_parent(parent)
    cm = read_json(calibration/'cal01_manifest.json')
    cs = read_json(calibration/'cal01_summary.json')
    ca = read_json(calibration/'cal01_acceptance.json')
    require(cs['status'] == ca['status'] == 'PASS' and all(ca['checks'].values()), 'CAL-01 is not ready')
    require(not cs['test_loaded'] and not cm['test_loaded'], 'Unexpected prior CAL-01 test access')
    require(cm['parent_manifest_sha256'] == sha(parent/'run_manifest.json'), 'CAL-01 has a different parent manifest')
    require(cm['parent_summary_sha256'] == spec['evidence_sha256']['exp05_pilot_summary.json'], 'CAL-01 parent summary mismatch')
    require(cm['parent_config_sha256'] == manifest['configuration_sha256'], 'CAL-01 configuration mismatch')
    require(cm['environment'] == manifest['environment'], 'CAL-01 runtime does not match EXP-05')
    for file, expected in CAL_CODE.items():
        require(sha(root/'src/ipat/experiments'/file) == expected, 'Frozen CAL-01 module changed: '+file)
    require(cm['cal01_code_sha256'] == {Path(k).stem:v for k,v in CAL_CODE.items()}, 'CAL-01 recorded source differs')
    torch.set_num_threads(manifest['environment']['cpu_threads'])
    device = configure_device(manifest['environment']['device'])
    require(environment(device) == manifest['environment'], 'Use the accepted EXP-05 runtime/device/threads; do not upgrade')
    config = manifest['config']
    state = read_json(cache/'preprocessing.json')
    require(digest(state) == config['preprocessing_logical_sha256'], 'Preprocessing changed')
    # Training cache is read only to hash its frozen content and audit metadata, never predicted/fitted.
    train = PackedOffers.load(cache/'train.npz')
    require(logical_hash(train.arrays) == config['logical_hashes']['train'], 'Training cache changed')
    train_meta = metadata(train)
    require(len(train) == 19071 and int(train.arrays['y'].sum()) == 13234, 'Training dummy provenance changed')
    del train
    val = load_partition(cache, 'validation', config['logical_hashes']['validation'])
    cal = load_partition(cache, 'calibration', config['logical_hashes']['calibration'])
    require(len(val)==7871 and len(cal)==6117, 'Parent cohort counts changed')
    disjoint({'train':train_meta,'validation':metadata(val),'calibration':metadata(cal)})
    expected_pairs = {(a,s) for a in ARMS for s in SEEDS}
    require(len(cs['runs'])==18 and {(r['arm'],r['seed']) for r in cs['runs']}==expected_pairs, 'Incomplete CAL-01 models')
    calrows = {(r['arm'],r['seed']):r for r in cs['runs']}
    verified = []
    for row in summary['runs']:
        arm, seed = row['arm'], row['seed']
        d, cd = parent/f'{arm}_{seed}', calibration/f'{arm}_{seed}'
        checkpoint = d/('selected.joblib' if arm=='GBDT' else 'selected.pt')
        entry = read_json(cd/'temperature.json')
        t = validate_temperature(entry, calrows[arm,seed], checkpoint)
        saved_val = load_npz(d/'predictions.npz')
        aligned(saved_val, metadata(val))
        require(abs(metrics(val.arrays['y'],saved_val['logits'])['nll']-row['selected']['nll'])<=1e-10, 'Parent validation score changed')
        require(sha(cd/'calibration_predictions.npz') == entry['prediction_sha256'], 'CAL-01 prediction hash mismatch')
        saved_cal = load_npz(cd/'calibration_predictions.npz')
        aligned(saved_cal, metadata(cal))
        from .cal01_core import scale_logits
        require(np.array_equal(scale_logits(saved_cal['raw_logits'],t), saved_cal['scaled_logits']), 'Saved CAL-01 temperature mismatch')
        raw, checks = evaluate_checkpoint(d,arm,seed,state,config,digest(manifest),row.get('selected_epoch'),val,cal,device,saved_val['logits'])
        checks['calibration_replay_max_logit_error'] = replay(raw,saved_cal['raw_logits'],'calibration')
        require(checks['checkpoint_sha256']==entry['checks']['checkpoint_sha256'], 'Checkpoint changed during replay')
        verified.append(dict(arm=arm,seed=seed,temperature=t,selected_epoch=row.get('selected_epoch'),
            checkpoint_sha256=sha(checkpoint),temperature_sha256=sha(cd/'temperature.json'),parent_checks=checks))
        print(f'Parent replay verified: {arm}/{seed}',flush=True)
    return manifest,state,device,verified,{'train':train_meta,'validation':metadata(val),'calibration':metadata(cal)}


def test_inference(directory, arm, seed, state, config, identity, selected_epoch, data, device, checkpoint_sha):
    """Reload one frozen predictor; inference twice, no optimisers or calibration solver."""
    from .exp05 import make_model, load_checkpoint, predict, tensor_hash
    from .exp05_references import prefix_summaries
    from ipat.data.bpi17 import logical_hash
    directory = Path(directory)
    file = directory/('selected.joblib' if arm=='GBDT' else 'selected.pt')
    require(sha(file)==checkpoint_sha, 'Test checkpoint hash mismatch')
    input_hash = logical_hash(data.arrays)
    if arm == 'GBDT':
        import joblib
        model = joblib.load(file)
        require(list(model.classes_)==[0,1], 'Tree class order changed')
        features = prefix_summaries(data,state)
        before = hashlib_joblib(model)
        raw = model.decision_function(features)
        again = model.decision_function(features)
        require(before==hashlib_joblib(model), 'Tree state changed during prediction')
    else:
        payload = load_checkpoint(file)
        require(payload['identity']==identity and payload['epoch']==selected_epoch, 'Selected checkpoint identity/epoch mismatch')
        model = make_model(arm,state,seed,config['model']).to(device)
        model.load_state_dict(payload['model'],strict=True)
        model.requires_grad_(False)
        model.eval()
        before = tensor_hash(model.state_dict())
        raw,_ = predict(model,data,device,config['microbatch'])
        again,_ = predict(model,data,device,config['microbatch'])
        require(before==tensor_hash(model.state_dict()), 'Neural state changed during prediction')
    require(np.array_equal(raw,again), 'Repeated test inference differs')
    require(logical_hash(data.arrays)==input_hash and sha(file)==checkpoint_sha, 'Inputs/checkpoint modified')
    return np.asarray(raw,dtype=np.float64), dict(exact_test_replay=True,model_state_unchanged=True,
        checkpoint_file_unchanged=True,input_arrays_unchanged=True)


def hashlib_joblib(model):
    import joblib
    return joblib.hash(model)


def input_files(parent, calibration, cache):
    paths = [parent/n for n in ('run_manifest.json','exp05_pilot_summary.json','exp05_acceptance.json')]
    paths += [calibration/n for n in ('cal01_manifest.json','cal01_summary.json','cal01_acceptance.json')]
    paths += [cache/n for n in ('preprocessing.json','train.npz','validation.npz','calibration.npz','test.npz')]
    for arm in ARMS:
        for seed in SEEDS:
            d,cd=parent/f'{arm}_{seed}',calibration/f'{arm}_{seed}'
            paths += [d/n for n in ('result.json','predictions.npz','selected.joblib' if arm=='GBDT' else 'selected.pt')]
            paths += [cd/'temperature.json',cd/'calibration_predictions.npz']
    return paths


def finish_reports(output, *, fixture_only=False):
    from .exp06_plots import plots
    result = aggregate(output,fixture_only=fixture_only)
    figures = plots(output)
    require(all((Path(output)/'plots'/f'{name}.{ext}').is_file() for name,_ in figures for ext in ('png','pdf','svg')), 'Missing figure export')
    require((Path(output)/'plots/report.html').is_file(), 'Missing HTML report')
    result['status'] = 'PASS'
    write_json(Path(output)/'exp06_summary.json', result)
    return result


def run(parent, calibration, cache, output, verification):
    from .exp05 import digest
    parent,calibration,cache,output = [Path(p).resolve() for p in (parent,calibration,cache,output)]
    require(not output.exists(), 'Use a new timestamped output directory; existing attempts are retained')
    for src in (parent,calibration,cache):
        require(src != output and src not in output.parents and output not in src.parents, 'Output overlaps an input directory')
    output.mkdir(parents=True,exist_ok=False)
    phase = 'preflight'
    try:
        verify_local_receipt(verification)
        inputs = {str(p):sha(p) for p in input_files(parent,calibration,cache)}
        modules = code_hashes()
        manifest,state,device,rows,partitions = preflight(parent,calibration,cache)
        # Include frozen source modules in the before/after inventory as well as data/checkpoints.
        for module in manifest['environment']['code_sha256']:
            p=Path(importlib.import_module(module).__file__).resolve()
            inputs[str(p)]=sha(p)
        for file in CAL_CODE:
            p=project_root()/'src/ipat/experiments'/file
            inputs[str(p)]=sha(p)
        receipt = dict(version='EXP06-v0.1',scope=SCOPE,protocol_accepted='2026-09-21',
            parent=str(parent),calibration=str(calibration),cache=str(cache),environment=manifest['environment'],
            code_sha256=modules,input_sha256=inputs,verification_sha256=sha(verification),
            predictive_partitions=['validation_replay','calibration_replay','test'],train_role='Hash and ID audit only',
            training=False,temperature_fitting=False,threshold_fitting=False,checkpoint_selection=False,
            bootstrap_draws=2000,bootstrap_seed=60601,primary_contrast='raw:T11-T00',frozen_runs=rows)
        write_json(output/'exp06_manifest.json',receipt)
        phase='test_inference'
        test = load_partition(cache,'test',manifest['config']['logical_hashes']['test'])
        meta = metadata(test)
        require(len(test)==8613, 'Expected 8,613 test offers')
        disjoint(dict(partitions,test=meta))
        tail = meta['lengths']>=41
        require(int(tail.sum())==243 and int((meta['y'][tail]==0).sum())==94 and int((meta['y'][tail]==1).sum())==149, 'Test tail differs from accepted contract')
        for row in rows:
            arm,seed = row['arm'],row['seed']
            started=time.perf_counter()
            raw,checks=test_inference(parent/f'{arm}_{seed}',arm,seed,state,manifest['config'],digest(manifest),row['selected_epoch'],test,device,row['checkpoint_sha256'])
            a=prediction_record(meta,raw,row['temperature'])
            d=output/f'{arm}_{seed}';d.mkdir()
            save_npz(d/'test_predictions.npz',**a)
            record=dict(row,**evaluate_record(a),checks=checks,elapsed_seconds=time.perf_counter()-started,
                        prediction_sha256=sha(d/'test_predictions.npz'))
            write_json(d/'result.json',record)
            print(f"Test evaluated: {arm}/{seed}; raw NLL={record['raw']['nll']:.6f}; scaled NLL={record['scaled']['nll']:.6f}",flush=True)
        phase='reports'
        finish_reports(output)
        require(inputs=={p:sha(p) for p in inputs}, 'Frozen input file changed during run')
        require(modules==code_hashes(), 'EXP-06 source changed during run')
        checks=dict(protocol_bound=True,parents_bound=True,cohort_integrity=True,frozen_execution=True,
            replay_verified=True,all_18_pairs=True,numerical_validity=True,reports_complete=True,inputs_unchanged=True)
        outputs={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='console.txt'}
        write_json(output/'checksums.json',dict(inputs=inputs,outputs=outputs))
        acceptance=dict(status='PASS',scope='Execution validity; scientific benefit is assessed separately',checks=checks,
            summary_sha256=sha(output/'exp06_summary.json'),manifest_sha256=sha(output/'exp06_manifest.json'),
            checksums_sha256=sha(output/'checksums.json'))
        write_json(output/'exp06_acceptance.json',acceptance)
        return acceptance
    except BaseException as error:
        # Preserve evidence even for interrupts. Never turn a report/IO failure into PASS.
        try:
            if (output/'exp06_summary.json').exists():
                s=read_json(output/'exp06_summary.json');s['status']='FAILED';write_json(output/'exp06_summary.json',s)
            write_json(output/'exp06_failure.json',dict(status='FAILED',phase=phase,error_type=type(error).__name__,error=str(error)))
        except Exception:
            pass
        raise
