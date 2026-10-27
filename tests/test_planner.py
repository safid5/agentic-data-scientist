import pytest
from agents.planner import (
    create_plan,
    small_dataset_template,
    large_dataset_template,
    standard_template,
    imbalanced_dataset_plan,
    high_dimensional_plan,
    select_preprocessing_strategy
)

def base_profile():
    return{
        "shape": {"rows": 1000, "cols": 10},
        "feature_types": {"numeric": ["a", "b"], "categorical": ["c"]},
        "missing_pct": {"a": 0.0, "b": 0.0, "c": 0.0},
        "imbalance_ratio": 1.0,
        "is_classification": True,
        "notes": []
    }
def test_small_dataset_plan():
    profile = base_profile()
    profile["shape"]["rows"] = 50
    plan = create_plan(profile)
    assert "select_models_small_data" in plan

def test_large_dataset_plan():
    profile = base_profile()
    profile["shape"]["rows"] = 500000
    plan = create_plan(profile)
    assert "select_models_large_data" in plan

def test_standard_dataset_plan():
    profile = base_profile()
    plan = create_plan(profile)
    assert "select_models" in plan
    assert "train_models" in plan
    assert "reflect" in plan

def test_imbalance_strategy_added():
    profile = base_profile()
    profile["imbalance_ratio"] = 5.0
    plan = create_plan(profile)
    assert isinstance(plan, list)

def test_missingness_triggers_imputation():
    profile = base_profile()
    profile["missing_pct"] = {"a" : 0.25}
    plan = create_plan(profile)
    assert "impute_missing_data" in plan

def test_high_dimensionality_triggers_pca():
    profile = base_profile()
    profile["shape"] = {"rows": 50, "cols": 200}
    plan = create_plan(profile)
    assert isinstance(plan, list)

def test_moderate_imbalance_adds_class_weights():
    profile = base_profile()
    profile["imbalance_ratio"] = 3.0
    plan = create_plan(profile)
    assert "set_class_weights" in plan

def test_severe_imbalance_adds_smote():
    profile = base_profile()
    profile["imbalance_ratio"] = 10.0
    plan = create_plan(profile)
    assert "apply_smote_oversampling" in plan
    assert "set_class_weights" in plan

def test_severe_missingness_triggers_severe_handler():
    profile = base_profile()
    profile["missing_pct"] = {"a": 0.60}
    plan = create_plan(profile)
    assert "handle_severe_missing_data" in plan

def test_high_dimensionality_triggers_feature_selection():
    profile = base_profile()
    profile["shape"] = {"rows": 1000, "cols": 60}
    plan = create_plan(profile)
    assert "apply_feature_selection" in plan

def test_very_high_dimensionality_triggers_pca():
    profile = base_profile()
    profile["shape"] = {"rows": 1000, "cols": 250}
    plan = create_plan(profile)
    assert "apply_pca_reduction" in plan
    assert "apply_feature_selection" in plan

def test_select_preprocessing_strategy_returns_dict():
    profile = base_profile()
    strategy = select_preprocessing_strategy(profile)
    assert isinstance(strategy, dict)
    assert "imputation" in strategy
    assert "encoding" in strategy
    assert "scaling" in strategy


def test_select_preprocessing_strategy_high_missing():
    profile = base_profile()
    profile["missing_pct"] = {"a": 0.40}
    strategy = select_preprocessing_strategy(profile)
    assert strategy["imputation"] == "iterative"

def test_plan_always_ends_with_fallback():
    profile = base_profile()
    plan = create_plan(profile)
    assert plan[-1] == "fallback_strategy"

def test_memory_hint_prioritises_model():
    profile = base_profile()
    hint = {"best_model": "RandomForest", "failed_steps": [], "replan_triggered": False}
    plan = create_plan(profile, memory_hint=hint)
    assert "prioritise_model:RandomForest" in plan

def test_memory_hint_none_does_not_crash():
    profile = base_profile()
    plan = create_plan(profile, memory_hint=None)
    assert isinstance(plan, list)

