from __future__ import annotations

import json

import numpy as np
import torch

from ipat.data.synthetic import IPATSyntheticConfig, generate_split
from ipat.experiments.exp01 import (
    Exp01Config,
    build_matched_models,
    deterministic_subset_indices,
    run_exp01,
    save_result,
)
from ipat.training.preprocessing import fit_baseline_preprocessing_stats


def tiny_split():
    cfg = IPATSyntheticConfig(
        master_seed=90903,
        n_train=64,
        n_val=16,
        n_cal=16,
        n_test=16,
        t_min=24,
        t_max=28,
    )
    return generate_split(cfg, "train", n_sequences=64)


def test_subset_indices_are_deterministic_unique_and_matched():
    a = deterministic_subset_indices(100, 32, 44)
    b = deterministic_subset_indices(100, 32, 44)
    assert a == b
    assert len(a) == len(set(a)) == 32
    assert a == sorted(a)


def test_matched_models_have_exact_shared_initial_state():
    split = tiny_split()
    stats = fit_baseline_preprocessing_stats(split)
    cfg = Exp01Config(subset_size=8, model_seed=12, dropout_override=0.0)
    baseline, retention, meta = build_matched_models(stats, cfg)
    b = baseline.state_dict()
    r = retention.state_dict()
    for name, value in b.items():
        if name in r and value.shape == r[name].shape:
            assert torch.equal(value, r[name]), name
    assert meta["max_initial_shared_state_abs_diff"] == 0.0
    assert meta["retention_extra_parameter_count"] == 41


def test_only_retention_scorer_is_extra_trainable_state():
    split = tiny_split()
    stats = fit_baseline_preprocessing_stats(split)
    _, _, meta = build_matched_models(stats, Exp01Config(subset_size=8))
    assert meta["retention_only_trainable_parameters"] == [
        "retention_scorer.linear.bias",
        "retention_scorer.linear.weight",
    ]


def test_zero_effect_initial_parity_and_matched_protocol_cpu():
    split = tiny_split()
    cfg = Exp01Config(
        subset_size=8,
        subset_seed=1,
        model_seed=2,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        dropout_override=0.0,
        gamma=1.0,
        target_loss=1e-12,
        target_accuracy=1.0,
        deterministic_algorithms=True,
        require_canonical_train_split=False,
    )
    result = run_exp01(split, cfg, device="cpu")
    assert result["initial_zero_effect_parity"]["within_tolerance"] is True
    assert result["matching"]["max_initial_shared_state_abs_diff"] == 0.0
    assert len(result["subset_indices"]) == 8
    assert result["baseline"]["trajectory"][0]["step"] == 1
    assert result["retention"]["trajectory"][0]["step"] == 1


def test_same_config_reproduces_one_step_results_cpu():
    split = tiny_split()
    cfg = Exp01Config(
        subset_size=8,
        subset_seed=3,
        model_seed=4,
        steps=1,
        learning_rate=1e-3,
        dropout_override=0.0,
        target_loss=1e-12,
        target_accuracy=1.0,
        require_canonical_train_split=False,
    )
    a = run_exp01(split, cfg, device="cpu")
    b = run_exp01(split, cfg, device="cpu")
    assert a["subset_indices"] == b["subset_indices"]
    assert a["baseline"]["first_loss"] == b["baseline"]["first_loss"]
    assert a["retention"]["first_loss"] == b["retention"]["first_loss"]


def test_initial_first_loss_is_matched_before_parameter_update():
    split = tiny_split()
    cfg = Exp01Config(
        subset_size=8,
        subset_seed=5,
        model_seed=6,
        steps=1,
        dropout_override=0.0,
        target_loss=1e-12,
        target_accuracy=1.0,
        require_canonical_train_split=False,
    )
    result = run_exp01(split, cfg, device="cpu")
    assert abs(result["baseline"]["first_loss"] - result["retention"]["first_loss"]) <= cfg.initial_parity_atol


def test_result_roundtrip_json(tmp_path):
    split = tiny_split()
    cfg = Exp01Config(
        subset_size=4,
        steps=1,
        target_loss=1e-12,
        target_accuracy=1.0,
        require_canonical_train_split=False,
    )
    result = run_exp01(split, cfg, device="cpu")
    path = save_result(result, tmp_path / "exp01.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == "EXP-01"
    assert loaded["interpretation_scope"].startswith("Implementation/trainability")
    assert loaded["initial_zero_effect_parity"]["within_tolerance"] is True
