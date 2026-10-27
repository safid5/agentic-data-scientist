import pytest
from agents.reflector import reflect, should_replan, apply_replan_strategy

def dummy_profile():
    return {
        "shape": {"rows": 100, "cols": 10},
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
        "missing_pct": {"a": 0.1, "b": 0.1},
        "imbalance ratio": 1.0,
        "is_classification": True,
        "notes": []
    }

def test_reflector_detect_low_f1():
    profile = dummy_profile()
    evaluation = {"model": "XGBoost", "balanced_accuracy": 0.5, "f1_macro": 0.4}
    all_metrics = [evaluation]
    result = reflect(profile, evaluation, all_metrics)
    assert result["status"] == "needs_attention"

def test_reflector_detect_low_overfitting():
    profile = dummy_profile()
    evaluation = {
        "model": "RF",
        "balanced_accuracy": 0.8,
        "f1_macro": 0.7,
        "train_accuracy": 0.99,
        "test_accuracy": 0.70,
    }
    result = reflect(profile, evaluation, [evaluation])
    assert isinstance(result["issues"], list)

def test_replan_trigger():
    profile = dummy_profile()
    evaluation = {"model": "RF", "balanced_accuracy": 0.4, "f1_macro": 0.3}
    result = reflect(profile, evaluation, [evaluation])
    assert isinstance(result["replan_recommended"], bool)

def test_apply_replan_strategy_add_steps():
    profile = dummy_profile()
    plan = ["profile_dataset", "build_preprocessor", "select_models", "train_model"]
    reflection = {"issues": ["overfitting detected"]}
    new_plan, new_profile = apply_replan_strategy(plan, profile, reflection)
    assert any("regularisation" in step.lower() for step in new_plan)