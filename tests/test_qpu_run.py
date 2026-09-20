"""Unit test for the QPU runner: integer-keyed samples + cardinality metadata (Q10)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("selected, expected_calls", [([0, 2], 1), ([1], 3)])
def test_qpu_runner_integer_samples_and_cardinality(monkeypatch, caplog, selected, expected_calls):
    """Exercise main without cloud packages, network access or filesystem writes."""
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock, mock_open

    import scripts.qpu_run as runner

    cloud = ModuleType("dwave.cloud")
    system = ModuleType("dwave.system")
    cloud.Client = MagicMock()
    cloud.Client.from_config.return_value.__enter__.return_value.get_solvers.return_value = []
    sampler = MagicMock()
    sampler.sample_qubo.return_value = SimpleNamespace(
        first=SimpleNamespace(sample={i: int(i in selected) for i in range(3)})
    )
    system.LeapHybridSampler = MagicMock(return_value=sampler)
    monkeypatch.setitem(sys.modules, "dwave.cloud", cloud)
    monkeypatch.setitem(sys.modules, "dwave.system", system)
    monkeypatch.setenv("DWAVE_API_TOKEN", "unit-test-not-a-credential")
    monkeypatch.setattr(runner, "load_config", lambda _: {"selection": {"k": 2}})
    monkeypatch.setattr(
        runner,
        "run_features",
        lambda _: pd.DataFrame(
            {"f0": [0.1, 0.2], "f1": [0.2, 0.3], "f2": [0.3, 0.4], "label": [0, 1]}
        ),
    )
    # QUBO coefficients are tuple-keyed; the returned sample is integer-keyed.
    monkeypatch.setattr(
        runner, "build_qubo", lambda *args: {(0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0, (0, 2): 0.5}
    )
    monkeypatch.setattr(runner, "ROOT", MagicMock())
    output = mock_open()
    monkeypatch.setattr(runner, "open", output, raising=False)

    assert runner.main() == 0
    result = json.loads("".join(call.args[0] for call in output().write.call_args_list))
    assert result["selected"] == selected
    assert result["requested_k"] == 2
    assert result["n_features_selected"] == len(selected)
    assert result["cardinality_ok"] is (len(selected) == 2)
    assert sampler.sample_qubo.call_count == expected_calls
    assert ("not directly comparable" in caplog.text) is (len(selected) != 2)
