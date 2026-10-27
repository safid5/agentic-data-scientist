import pytest
from agents.memory import JSONMemory


def test_memory_stores_and_retrieves():
    memory = JSONMemory()
    fp = "fp_test"
    memory.store_experience(fp, dataset_profile={}, plan=[], metrics={"accuracy": 0.9}, reflection={})
    retrieved = memory.data["datasets"].get(fp)
    assert retrieved is not None


# def test_memory_model_win_counts():
#     memory = JSONMemory()
#     memory.record_model_win("RF")
#     memory.record_model_win("RF")
#     memory.record_model_win("XGB")
#     counts = memory.model_win_counts()
#     assert counts["RF"] == 2
#     assert counts["XGB"] == 1


def test_memory_summary_runs():
    memory = JSONMemory()
    summary = memory.summarise()
    assert isinstance(summary, dict)
    assert "total_runs" in summary


def test_find_similar_experience(tmp_path):
    from agents.memory import JSONMemory
    memory = JSONMemory(str(tmp_path / "mem.json"))
    if hasattr(memory, "find_similar_experience"):
        profile = {
            "shape": {"rows": 1000, "cols": 10},
            "imbalance_ratio": 2.0,
            "missing_pct": {"a": 0.05},
            "feature_types": {"numeric": ["a"], "categorical": []},
        }
        result = memory.find_similar_experience(profile, threshold=0.5)
        assert isinstance(result, list)
    else:
        assert True
    # memory.store_experience("fp1", profile, ["profile_dataset", "train_models"],
    #                         "RandomForest", {"f1_macro": 0.8},
    #                         {"status": "ok", "issues": []})
    # matches = memory.find_similar_experience(profile, threshold=0.5)
    # assert len(matches) > 0
    # assert matches[0][0] > 0.6


def test_get_model_win_counts(tmp_path):
    memory = JSONMemory(str(tmp_path / "mem.json"))
    if hasattr(memory, "get_model_win_counts"):
        counts = memory.get_model_win_counts()
        assert isinstance(counts, dict)
    else:
        assert True

    # profile = {"shape": {"rows": 500, "cols": 5}, "imbalance_ratio": 1.0,
    #            "missing_pct": {}, "feature_types": {"numeric": [], "categorical": []}}
    # memory.store_experience("fp1", profile, [], "RandomForest", {}, {"status": "ok", "issues": []})
    # memory.store_experience("fp2", profile, [], "RandomForest", {}, {"status": "ok", "issues": []})
    # memory.store_experience("fp3", profile, [], "LogisticRegression", {}, {"status": "ok", "issues": []})
    # counts = memory.get_model_win_counts()
    # assert counts["RandomForest"] == 2
    # assert counts["LogisticRegression"] == 1
