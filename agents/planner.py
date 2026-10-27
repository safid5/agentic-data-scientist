"""
Planner Agent - Students must extend this significantly

The planner analyzes dataset characteristics and generates an execution plan.
Your task is to implement sophisticated planning logic that adapts to different
dataset types, sizes, and characteristics.

TODO: Extend this module with:
1. Sophisticated planning logic based on dataset profiles
2. Different plan templates for different scenarios
3. Memory-guided planning (use past successful strategies)
4. Dependency management (task ordering)
5. Conditional planning (if X then Y else Z)
6. Fallback strategies for edge cases
"""

from typing import Any, Dict, List, Optional


def create_plan(
        dataset_profile: Dict[str, Any],
        memory_hint: Optional[Dict[str, Any]] = None) -> List[str]:
    # dataset characteristics
    rows = dataset_profile["shape"]["rows"]
    cols = dataset_profile["shape"]["cols"]
    numeric = dataset_profile["feature_types"]["numeric"]
    categorical = dataset_profile["feature_types"]["categorical"]
    missing_pct = dataset_profile["missing_pct"]
    imbalance_ratio = dataset_profile.get("imbalance_ratio", 1.0)
    is_classification = dataset_profile["is_classification"]

    plan = []

    """
    Generate an execution plan based on dataset characteristics.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dictionary containing dataset metadata including:
            - shape: {rows: int, cols: int}
            - feature_types: {numeric: List[str], categorical: List[str]}
            - imbalance_ratio: float (majority/minority class ratio)
            - missing_pct: Dict[str, float] (missing % per column)
            - is_classification: bool
            - notes: List[str] (warnings/observations)
        memory_hint: Optional dict with info from previous runs on similar datasets
    
    Returns:
        List of task names representing the execution plan
        
    Example:
        >>> profile = {"shape": {"rows": 5000}, "imbalance_ratio": 4.5}
        >>> plan = create_plan(profile)
        >>> print(plan)
        ['profile_dataset', 'consider_imbalance_strategy', 'train_models', ...]
    
    TODO for students:
    - Implement conditional logic based on dataset size
    - Add different strategies for imbalanced datasets
    - Handle high-cardinality categorical features
    - Use memory hints to prioritize successful models
    - Create plan templates for common scenarios
    - Add preprocessing steps based on data quality
    
    "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    """
    imb = dataset_profile.get("imbalance_ratio") or 1.0

    # Basic plan structure (students should make this much more sophisticated)
    if rows < 800:
        plan = small_dataset_template()
    elif rows > 20000:
        plan = large_dataset_template()
    elif imb >= 3.0 and cols > 50:
        plan = high_dimensional_plan()
    elif imb >= 3.0:
        plan = imbalanced_dataset_plan()
    elif cols > 50:
        plan = high_dimensional_plan()
    else:
        plan = standard_template()

    # TODO: Add sophisticated logic here
    # Example: Check for imbalance
    if imb >= 3.0:
        # TODO: Make this more sophisticated
        # Consider: SMOTE, class weights, threshold tuning, etc.
        plan.insert(plan.index("train_models"), "apply_smote_oversampling")
        plan.insert(plan.index("train_models"), "set_class_weights")
    elif imb >= 2.0:
        plan.insert(plan.index("train_models"), "set_class_weights")

    # TODO: Add logic for small datasets
    if dataset_profile["shape"]["rows"] < 5000:
        plan.append("apply_regularization")
    elif dataset_profile["shape"]["cols"] < 10000:
        plan.append("apply_data_sampling")

    # TODO: Add logic for high-cardinality categoricals
    # high_card_cats = [c for c in categorical_cols if n_unique[c] > 50]
    high_card = dataset_profile.get("high_cardinality_cols", [])
    if high_card:
        plan.insert(plan.index("build_preprocessor"), "apply_target_encoding")

    # TODO: Use memory hints
    if memory_hint and memory_hint.get("best_model"):
        best = memory_hint["best_model"]
        select_stop = next(
            (s for s in plan if s.startswith("select_models")), None
        )
        if select_stop:
            plan.insert(plan.index("select_models") + 1, f"prioritise_model:{best}")

    # TODO: Add logic based on missing data
    max_missing = max(missing_pct.values()) if missing_pct else 0

    if max_missing > 0.30:
        plan.insert(1, "handle_severe_missing_data")
    elif max_missing > 0.10:
        plan.insert(1, "impute_missing_data")

    # select preprocessing strategies
    preprocessing_strategy = select_preprocessing_strategy(dataset_profile)


    plan.append("fallback_strategy")
    return plan






# TODO: Add helper functions for planning
def small_dataset_template():
    return [
        "profile_dataset",
        "build_preprocessor",
        "select_models_small_data",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def large_dataset_template():
    return [
        "profile_dataset",
        "build_preprocessor",
        "select_models_large_data",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def standard_template():
    return [
        "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def imbalanced_dataset_plan():
    return [
        "profile_dataset",
        "build_preprocessor",
        "set_class_weights",
        "apply_smote_oversampling",
        "select_models",
        "train_models",
        "evaluate_with_f1",
        "reflect",
        "write_report",
    ]


def high_dimensional_plan():
    return [
        "profile_dataset",
        "build_preprocessor",
        "apply_feature_selection",
        "apply_pca_reduction",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def select_preprocessing_strategy(dataset_profile: Dict[str, Any]) -> Dict[str, str]:
    strategies = {}

    missing_pct = dataset_profile.get("missing_pct", {})
    max_missing = max(missing_pct.values()) if missing_pct else 0
    high_card = dataset_profile.get("high_cardinality_cols", [])
    cols = dataset_profile["shape"]["cols"]
    cat_cols = dataset_profile.get("feature_types", {}).get("categorical", [])

    # imputation strategy
    if max_missing > 0.30:
        strategies["imputation"] = "iterative"
    elif max_missing > 0.10:
        strategies["imputation"] = "median"
    else:
        strategies["imputation"] = "simple"

    # encoding strategy
    if high_card:
        strategies["encoding"] = "target_encoding"
    elif cat_cols:
        strategies["encoding"] = "onehot"
    else:
        strategies["encoding"] = "none"

    # scaling strategy
    if cols > 50:
        strategies["scaling"] = "standard"
    else:
        strategies["scaling"] = "minmax"
    return strategies

""""
def estimate_plan_cost(plan: List[str]) -> int:  # For cost-aware planning
    step_costs = {
        "apply_smote_oversampling": 3,
        "apply_pca_reduction": 2,
        "apply_iterative_imputation": 2,
        "apply_feature_selection": 2,
        "train_models": 2,
        "apply_data_sampling": 1,
        "apply_regularization": 1,
        "handle_severe_missing_data": 2,
        "apply_target_encoding": 1,
        "set_class_weights": 1,
        "apply_feature_engineering": 2,
        "apply_ensemble_methods": 3,
    }

    return sum(step_costs.get(step, 1) for step in plan)
"""
