import os
import tempfile
import pandas as pd
import pytest

from agentic_data_scientist import AgenticDataScientist

def test_agentic_instantiation():
    agent = AgenticDataScientist(verbose = False)
    assert agent is not None
    assert hasattr(agent, "memory")

def test_load_data():
    agent = AgenticDataScientist(verbose = False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        pd.DataFrame({"x": [1, 2], "label" : [0, 1]}).to_csv(path, index=False)

        df = agent.load_data(path)
        assert df.shape == (2, 2)

def test_load_data_missing():
    agent = AgenticDataScientist(verbose = False)
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "does_not_exist.csv")
        try:
            agent.load_data(missing_path)
        except FileNotFoundError:
            assert True
        else:
            assert False, "Expected FileNotFoundError"

def test_execute_with_retry_runs_function():
    agent = AgenticDataScientist(verbose = False)
    result = agent.execute_with_retry("dummy", lambda: 123)
    assert result == 123

def test_execute_with_retry_runs_function_handles_failure():
    agent = AgenticDataScientist(verbose = False)
    calls = {"count" : 0}

    def falling_fn():
        calls["count"] += 1
        raise ValueError("Fail")

    result = agent.execute_with_retry("dummy", falling_fn, max_retry = 2)
    assert result is None
    assert calls["count"] == 2

def test_should_execute_step_filters():
    agent = AgenticDataScientist(verbose = False)
    plan = ["train_models", "prioritise_model:RF", "fallback_for: LR", "fallback_strategy: tree"]
    assert agent.should_execute_step("train_models", plan) is True
    assert agent.should_execute_step("prioritise_model: RF", plan) is False
    assert agent.should_execute_step("fallback_for: LR", plan) is False
    assert agent.should_execute_step("fallback_strategy: tree", plan) is False

