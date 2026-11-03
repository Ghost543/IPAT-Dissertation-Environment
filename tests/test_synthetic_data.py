from __future__ import annotations

import numpy as np

from ipat.data.synthetic import (
    IPATSyntheticConfig,
    apply_event_deletion,
    apply_missingness_corruption,
    apply_numerical_noise,
    apply_temporal_jitter,
    compute_train_numeric_scale,
    generate_position_sweep,
    generate_split,
    load_hdf5,
    recompute_truth,
    save_hdf5,
    verify_inverse_feature_family,
)


def cfg() -> IPATSyntheticConfig:
    return IPATSyntheticConfig(
        master_seed=123456,
        n_train=256,
        n_val=64,
        n_cal=64,
        n_test=128,
        t_min=24,
        t_max=32,
    )


def assert_same(a, b, names=None):
    names = names or a.arrays.keys()
    for name in names:
        assert name in b.arrays
        assert np.array_equal(a[name], b[name]), name


def test_reproducible_same_seed():
    c = cfg()
    a = generate_split(c, "train", n_sequences=32)
    b = generate_split(c, "train", n_sequences=32)
    assert_same(a, b)


def test_independent_split_ids():
    c = cfg()
    train = generate_split(c, "train", n_sequences=32)
    val = generate_split(c, "val", n_sequences=32)
    assert set(train["sequence_id"].tolist()).isdisjoint(set(val["sequence_id"].tolist()))


def test_lengths_and_exact_signal_count():
    c = cfg()
    data = generate_split(c, "train", n_sequences=64)
    assert np.all((data["length"] >= c.t_min) & (data["length"] <= c.t_max))
    assert np.all(data["signal_mask"].sum(axis=1) == c.n_signal)
    assert np.all(data["signal_mask"] <= data["valid_mask"])


def test_truth_recomputation():
    c = cfg()
    data = generate_split(c, "test", n_sequences=64)
    contribution, logits, p = recompute_truth(data, c)
    assert np.allclose(contribution, data["contribution"], atol=1e-12, rtol=0)
    assert np.allclose(logits, data["sequence_logit"], atol=1e-12, rtol=0)
    assert np.allclose(p, data["p_true"], atol=1e-12, rtol=0)


def test_label_rng_does_not_change_features():
    c = cfg()
    a = generate_split(c, "train", n_sequences=128, label_seed_offset=0)
    b = generate_split(c, "train", n_sequences=128, label_seed_offset=999)
    assert_same(a, b, [k for k in a.arrays if k != "y"])
    assert np.any(a["y"] != b["y"])


def test_inverse_candidate_family():
    data = generate_split(cfg(), "train", n_sequences=32)
    assert max(verify_inverse_feature_family(data).values()) < 1e-12


def test_observed_zero_and_missing_zero_are_distinguishable():
    data = generate_split(cfg(), "train", n_sequences=256)
    valid = data["valid_mask"]
    observed_zero = valid & (~data["missing_mask"][..., 5]) & (data["x_obs"][..., 5] == 0.0)
    missing_zero = valid & data["missing_mask"][..., 5] & (data["x_obs"][..., 5] == 0.0)
    assert observed_zero.any()
    assert missing_zero.any()
    assert np.all(data["missing_mask"][..., 5][observed_zero] == 0)
    assert np.all(data["missing_mask"][..., 5][missing_zero] == 1)


def test_padding_is_separate_and_zeroed():
    data = generate_split(cfg(), "test", n_sequences=64)
    pad = ~data["valid_mask"]
    assert np.all(data["x_obs"][pad] == 0.0)
    assert np.all(data["contribution"][pad] == 0.0)
    assert np.all(data["signal_mask"][pad] == 0)
    assert np.all(data["event_type"][pad] == -1)
    assert np.all(data["position"][pad] == -1)


def test_expected_prevalence_is_near_balanced():
    data = generate_split(cfg(), "train", n_sequences=3000)
    assert abs(float(data["p_true"].mean()) - 0.5) < 0.035
    assert abs(float(data["y"].mean()) - 0.5) < 0.05


def test_position_sweep_preserves_target_and_label():
    data = generate_position_sweep(cfg(), n_triplets=40, seed=777)
    for k in range(40):
        rows = slice(3 * k, 3 * k + 3)
        assert np.ptp(data["p_true"][rows]) == 0.0
        assert np.ptp(data["sequence_logit"][rows]) == 0.0
        assert np.ptp(data["y"][rows]) == 0
        assert data["variant"][rows].tolist() == [b"early", b"middle", b"late"]


def test_missingness_corruption_rate_and_replay():
    clean = generate_split(cfg(), "test", n_sequences=256)
    a = apply_missingness_corruption(clean, target_rate=0.20, seed=55)
    b = apply_missingness_corruption(clean, target_rate=0.20, seed=55)
    assert_same(a, b)
    assert abs(a.metadata["corruption"]["achieved_rate"] - 0.20) < 0.02
    assert np.array_equal(clean["p_true"], a["p_true"])
    assert np.array_equal(clean["y"], a["y"])


def test_event_deletion_replay_and_shortens_sequences():
    clean = generate_split(cfg(), "test", n_sequences=64)
    a = apply_event_deletion(clean, fraction=0.25, seed=66)
    b = apply_event_deletion(clean, fraction=0.25, seed=66)
    assert_same(a, b)
    assert np.all(a["length"] < clean["length"])
    assert np.array_equal(clean["p_true"], a["p_true"])
    assert np.array_equal(clean["y"], a["y"])


def test_numerical_noise_uses_train_scale_and_replays():
    c = cfg()
    train = generate_split(c, "train", n_sequences=256)
    test = generate_split(c, "test", n_sequences=64)
    scale = compute_train_numeric_scale(train)
    a = apply_numerical_noise(test, q=0.10, train_scale=scale, seed=77)
    b = apply_numerical_noise(test, q=0.10, train_scale=scale, seed=77)
    assert_same(a, b)
    observed = np.broadcast_to(test["valid_mask"][..., None], test["x_obs"].shape) & (~test["missing_mask"])
    assert np.any(a["x_obs"][observed] != test["x_obs"][observed])
    assert np.array_equal(a["p_true"], test["p_true"])


def test_temporal_jitter_positive_ordered_and_replays():
    clean = generate_split(cfg(), "test", n_sequences=64)
    a = apply_temporal_jitter(clean, q=0.10, seed=88)
    b = apply_temporal_jitter(clean, q=0.10, seed=88)
    assert_same(a, b)
    for i, t_ in enumerate(a["length"]):
        t = int(t_)
        assert np.all(a["delta"][i, :t] > 0)
        assert np.all(np.diff(a["timestamp"][i, :t]) > 0)
    assert np.array_equal(a["p_true"], clean["p_true"])


def test_all_generated_values_are_finite():
    data = generate_split(cfg(), "train", n_sequences=128)
    for name, arr in data.arrays.items():
        if np.issubdtype(arr.dtype, np.floating):
            assert np.all(np.isfinite(arr)), name


def test_hdf5_roundtrip(tmp_path):
    data = generate_split(cfg(), "test", n_sequences=32)
    loaded = load_hdf5(save_hdf5(data, tmp_path / "sample.h5"))
    assert loaded.metadata == data.metadata
    assert_same(data, loaded)
