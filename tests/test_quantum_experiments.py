"""Standalone A2/A3/A4 tests, including real tiny Qiskit statevectors."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2

from quobo.quantum_experiments import (
    FeatureSplit,
    ablation_grid,
    aggregate_sampler_results,
    centered_target_alignment,
    ibm_kernel,
    load_split,
    make_statevector_kernel,
    overlap_circuits,
    prepare_split,
    run_ablation,
    sampler_kernel,
    smoke_split,
    train_alignment,
    train_kernel_experiment,
    write_new_json,
)

ROOT = Path(__file__).resolve().parents[1]


def require_quantum():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_machine_learning")


def test_review_regressions_for_ids_caps_and_atomic_write(tmp_path):
    split = smoke_split()
    values = vars(split).copy()
    values["groups_train"] = np.arange(len(split.y_train))
    values["groups_test"] = np.arange(len(split.y_test), dtype=float)
    with pytest.raises(ValueError, match="groups overlap"):
        FeatureSplit(**values)
    for cap in (3, 5):
        with pytest.raises(ValueError, match="number of classes"):
            prepare_split(split, 4, max_train=cap)
    data, _ = prepare_split(split, 4, max_train=6)
    assert set(np.unique(data["y_train"], return_counts=True)[1]) == {2}
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_new_json(path, {"bad": float("nan")})
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_sampler_retains_job_id_before_failed_wait():
    require_quantum()
    events = []

    class Job:
        def job_id(self):
            return "recoverable-job"

        def result(self):
            assert events == ["recoverable-job"]
            raise RuntimeError("temporary wait failure")

    sampler = SimpleNamespace(run=lambda *args, **kwargs: Job())
    with pytest.raises(RuntimeError, match="recoverable-job"):
        sampler_kernel(np.array([[0.1, 0.2]]), sampler=sampler, on_submitted=events.append)


def test_transpiled_self_overlap_keeps_encoding_gates():
    require_quantum()
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuits, _, _, _ = overlap_circuits(np.array([[0.1, 0.2]]))
    manager = generate_preset_pass_manager(
        backend=GenericBackendV2(2, seed=1), optimization_level=1, seed_transpiler=1
    )
    compiled = manager.run(circuits)[0]
    assert set(compiled.count_ops()) - {"measure", "barrier"}


def test_ibm_execute_cli_fails_closed_and_records_job(tmp_path, monkeypatch, capsys):
    """The --execute CLI never reads an echoed token, and persists the job ID
    before the wait so a failed retrieval cannot lose a paid submission."""
    import getpass

    from scripts import ibm_kernel as cli

    calls = {}

    class Service:
        def __init__(self, **kwargs):
            calls["service"] = kwargs

        def backend(self, name):
            calls["backend"] = name
            return GenericBackendV2(2, seed=1)

    def sampler_factory(*, mode):
        calls["mode"] = mode
        return StatevectorSampler(seed=2)

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime",
        SimpleNamespace(QiskitRuntimeService=Service, SamplerV2=sampler_factory),
    )

    def echoing(prompt=""):
        warnings.warn("problem in getpass", getpass.GetPassWarning)
        return "echoed-token"

    monkeypatch.setattr(cli.getpass, "getpass", echoing)
    npz = tmp_path / "x.npz"
    np.savez(npz, X=np.array([[0.1, 0.2], [0.7, 1.1]]))
    out = tmp_path / "result.json"
    # The CLI converts the insecure-terminal ValueError into a fail-closed
    # argparse exit (status 2) with the reason on stderr; no artifacts written.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--execute",
                "--input",
                str(npz),
                "--output",
                str(out),
                "--backend",
                "b",
                "--instance",
                "i",
                "--prompt-token",
            ]
        )
    assert excinfo.value.code == 2
    assert "refusing to read a token" in capsys.readouterr().err
    assert not out.exists() and not list(tmp_path.glob("*.job.json"))

    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "mock-token")
    rc = cli.main(
        [
            "--execute",
            "--input",
            str(npz),
            "--output",
            str(out),
            "--backend",
            "b",
            "--instance",
            "i",
            "--prompt-token",
            "--shots",
            "64",
        ]
    )
    assert rc == 0
    record = json.loads((tmp_path / "result.json.job.json").read_text())
    assert record["job_id"] and record["backend"] == "b"
    result = json.loads(out.read_text())
    assert result["job_id"] == record["job_id"] and result["network_used"]
    assert all("mock-token" not in p.read_text() for p in tmp_path.glob("*.json*"))


def test_multiclass_target_is_category_invariant():
    labels = np.array(["cat", "dog", "bird", "cat", "dog", "bird"])
    target = (labels[:, None] == labels).astype(float)
    assert centered_target_alignment(target, labels) == pytest.approx(1)
    assert centered_target_alignment(target, [300, -2, 8, 300, -2, 8]) == pytest.approx(1)
    assert centered_target_alignment(np.ones((6, 6)), labels) == 0
    with pytest.raises(ValueError, match="two classes"):
        centered_target_alignment(np.ones((2, 2)), ["a", "a"])


def test_group_identity_contract_and_pickle_free_split(tmp_path):
    split = smoke_split()
    values = vars(split).copy()
    values["groups_test"] = np.array([split.groups_train[0]] * len(split.y_test))
    with pytest.raises(ValueError, match="groups overlap"):
        FeatureSplit(**values)
    values = vars(split).copy()
    values["ids_test"] = np.array([split.ids_train[0]] * len(split.y_test))
    with pytest.raises(ValueError, match="IDs must be unique"):
        FeatureSplit(**values)
    values = vars(split).copy()
    values["transform_fit"] = "all_data"
    with pytest.raises(ValueError, match="raw or train_only"):
        FeatureSplit(**values)
    path = tmp_path / "split.npz"
    np.savez(path, **vars(split))
    assert load_split(path).fingerprint() == split.fingerprint()
    np.savez(path, X_train=split.X_train)
    with pytest.raises(ValueError, match="missing keys"):
        load_split(path)


def test_preprocessing_never_fits_test_and_preserves_ids():
    split = smoke_split()
    prepared, original = prepare_split(split, 4, max_train=9, max_test=4)
    split.X_test[:] = 1e8
    split.y_test[:] = "a"
    changed, provenance = prepare_split(split, 4, max_train=9, max_test=4)
    for key in ("selected", "mi_scores", "scaler_data_min", "scaler_data_max", "train_indices"):
        assert provenance[key] == original[key]
    np.testing.assert_array_equal(prepared["X_train"], changed["X_train"])
    assert provenance["ids_train"] == split.ids_train[provenance["train_indices"]].tolist()
    assert set(provenance["groups_train"]).isdisjoint(provenance["groups_test"])
    assert set(changed["y_train"]) == {"a", "b", "c"}
    assert changed["X_test"].min() > 1.52  # no fitting/clipping on held-out extrema
    with pytest.raises(ValueError, match="number of classes"):
        prepare_split(split, 4, max_train=2)


def test_real_trainable_kernel_depends_on_parameters_and_matches_fixed_at_one():
    require_quantum()
    X = np.array([[0.1, 0.2], [0.5, 0.8], [1.1, 1.3]])
    trained = make_statevector_kernel(2, trainable=True)
    fixed = make_statevector_kernel(2)
    trained.assign_training_parameters([1, 1])
    baseline = trained.evaluate(X)
    np.testing.assert_allclose(baseline, fixed.evaluate(X), atol=1e-10)
    trained.assign_training_parameters([0.3, 1.7])
    assert not np.allclose(baseline, trained.evaluate(X))


def test_tiny_real_statevector_learning_is_seeded_and_restores_best():
    require_quantum()
    split = smoke_split(features=2)
    data, _ = prepare_split(split, 2)
    kernel = make_statevector_kernel(2, trainable=True)
    result = train_alignment(kernel, data["X_train"], data["y_train"], iterations=3, seed=5)
    restored_alignment = centered_target_alignment(
        kernel.evaluate(data["X_train"]), data["y_train"]
    )
    assert restored_alignment == pytest.approx(result["best_alignment"])
    assert result["best_alignment"] >= result["initial_alignment"]
    assert result["evaluations"] == 10
    assert any(
        abs(row["loss"] - result["history"][0]["loss"]) > 1e-7 for row in result["history"][1:]
    )
    assert result["best_parameters"] != result["initial_parameters"]
    repeated = train_alignment(
        make_statevector_kernel(2, trainable=True),
        data["X_train"],
        data["y_train"],
        iterations=3,
        seed=5,
    )
    assert result == repeated
    experiment = train_kernel_experiment(split, budget=2, iterations=2)
    assert len(experiment["trained"]["predictions"]) == 6
    assert len(experiment["fixed"]["predictions"]) == 6
    assert experiment["trained"]["C_tuned"] is False


def test_best_restore_when_last_iterate_is_worse():
    class ToyKernel:
        def __init__(self):
            self.training_parameters = ["theta"]
            self.point = np.ones(1)

        def assign_training_parameters(self, point):
            self.point = np.asarray(point).copy()

        def evaluate(self, X):
            t = self.point[0]
            # Off-diagonal similarity peaks at t=.9; returned matrices remain PSD.
            return np.array(
                [
                    [1, np.exp(-(((t - 0.9) / 0.1) ** 2)), 0],
                    [np.exp(-(((t - 0.9) / 0.1) ** 2)), 1, 0],
                    [0, 0, 1],
                ]
            )

    kernel = ToyKernel()
    result = train_alignment(
        kernel, np.ones((3, 2)), np.array(["a", "a", "b"]), iterations=1, seed=1, learning_rate=3
    )
    assert result["history"][-1]["loss"] > -result["best_alignment"]
    np.testing.assert_array_equal(kernel.point, result["best_parameters"])


def test_ablation_dry_run_and_resume(tmp_path, monkeypatch):
    split = smoke_split()
    grid = ablation_grid()
    assert len(grid) == 24
    report = run_ablation(split, tmp_path)
    assert report["planned"] == 24 and report["completed"] == []
    assert len(list(tmp_path.glob("*.json"))) == 1
    require_quantum()
    grid = ablation_grid([1], ["linear"], [4])
    first = run_ablation(split, tmp_path, configs=grid, dry_run=False)
    assert len(first["completed"]) == 1
    result_path = tmp_path / f"{first['completed'][0]}.json"
    content = result_path.read_bytes()

    def fail(*args, **kwargs):
        raise AssertionError("resume must not recompute existing kernels")

    monkeypatch.setattr("quobo.quantum_experiments.make_statevector_kernel", fail)
    second = run_ablation(split, tmp_path, configs=grid, dry_run=False)
    assert second["completed"] == [] and second["reused"] == first["completed"]
    assert result_path.read_bytes() == content
    changed = run_ablation(split, tmp_path, configs=grid, seed=43)
    assert changed["manifest"] != first["manifest"]
    bad = json.loads(content)
    bad["specification"]["seed"] = -1
    result_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="inconsistent"):
        run_ablation(split, tmp_path, configs=grid)


def counts_result(counts):
    return SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts)))


def test_sampler_aggregation_no_forced_diagonal_or_psd():
    K, shots = aggregate_sampler_results(
        [
            counts_result({"00": 9, "11": 1}),
            counts_result({"00": 3, "01": 7}),
            counts_result({"00": 8, "10": 2}),
        ],
        [(0, 0), (0, 1), (1, 1)],
        (2, 2),
        True,
    )
    np.testing.assert_allclose(K, [[0.9, 0.3], [0.3, 0.8]])
    assert shots == [10, 10, 10]
    with pytest.raises(ValueError, match="count does not match"):
        aggregate_sampler_results([], [(0, 0)], (1, 1), True)
    with pytest.raises(ValueError, match="positive shots"):
        aggregate_sampler_results([counts_result({})], [(0, 0)], (1, 1), True)
    with pytest.raises(ValueError, match="binary-string"):
        aggregate_sampler_results([counts_result({"0x0": 10})], [(0, 0)], (1, 1), True)


def test_overlap_circuits_and_real_local_sampler():
    require_quantum()
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector

    X = np.array([[0.1, 0.2], [0.7, 1.1]])
    circuits, pairs, shape, symmetric = overlap_circuits(X)
    assert pairs == [(0, 0), (0, 1), (1, 1)] and shape == (2, 2) and symmetric
    exact = make_statevector_kernel(2).evaluate(X)
    for circuit, (i, j) in zip(circuits, pairs):
        assert Statevector.from_instruction(
            circuit.remove_final_measurements(inplace=False)
        ).probabilities()[0] == pytest.approx(exact[i, j])
    sampled = sampler_kernel(X, sampler=StatevectorSampler(seed=10), shots=4096)
    np.testing.assert_allclose(sampled["kernel"], exact, atol=0.04)
    rectangular = sampler_kernel(X[:1], X, sampler=StatevectorSampler(seed=10), shots=4096)
    assert np.asarray(rectangular["kernel"]).shape == (1, 2)
    assert rectangular["shots_observed"] == [4096, 4096]


def test_ibm_never_implicitly_executes_and_rejects_zne(monkeypatch):
    require_quantum()
    import builtins

    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("qiskit_ibm_runtime"):
            raise AssertionError("dry-run must not import Runtime or discover credentials")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    X = np.array([[0.1, 0.2], [0.7, 1.1]])
    plan = ibm_kernel(X)
    assert plan["network_used"] is False and plan["circuit_count"] == 3
    assert plan["seed_sampler"] is None
    with pytest.raises(ValueError, match="explicit token, backend, and instance"):
        ibm_kernel(X, execute=True)
    with pytest.raises(ValueError, match="ZNE"):
        ibm_kernel(X, mitigation="zne")


def test_ibm_execute_with_mock_runtime_and_installed_transpiler(monkeypatch):
    require_quantum()
    from qiskit.primitives import StatevectorSampler
    from qiskit.providers.fake_provider import GenericBackendV2

    calls = {}

    class Service:
        def __init__(self, **kwargs):
            calls["service"] = kwargs

        def backend(self, name):
            calls["backend"] = name
            return GenericBackendV2(2, seed=1)

    def sampler_factory(*, mode):
        calls["mode"] = mode
        return StatevectorSampler(seed=2)

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime",
        SimpleNamespace(QiskitRuntimeService=Service, SamplerV2=sampler_factory),
    )
    result = ibm_kernel(
        np.array([[0.1, 0.2], [0.7, 1.1]]),
        execute=True,
        token="dummy-not-a-secret",
        backend="mock_backend",
        instance="mock_instance",
        shots=64,
    )
    assert calls["service"]["token"] == "dummy-not-a-secret"
    assert calls["backend"] == "mock_backend"
    assert result["network_used"] and result["shots_observed"] == [64, 64, 64]
    assert "dummy-not-a-secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "script", ["train_quantum_kernel.py", "kernel_ablation.py", "ibm_kernel.py"]
)
def test_help_is_credential_free_and_safe(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout
    assert not list(tmp_path.iterdir())


def test_cli_dry_runs_and_no_overwrite(tmp_path):
    require_quantum()
    commands = [
        ("train_quantum_kernel.py", []),
        ("kernel_ablation.py", ["--output", str(tmp_path / "ablation")]),
        ("ibm_kernel.py", []),
    ]
    for script, extra in commands:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--dry-run", *extra],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
    path = tmp_path / "frozen.json"
    write_new_json(path, {"frozen": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_json(path, {"frozen": False})
    assert path.read_bytes() == before
    invalid = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ibm_kernel.py"), "--execute"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert invalid.returncode == 2 and "--prompt-token" in invalid.stderr
