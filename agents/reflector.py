"""
Reflector Agent - Students must extend this significantly

The reflector evaluates execution results, identifies issues, and suggests improvements.
Your task is to implement sophisticated analysis that goes beyond simple threshold checks.

TODO: Extend this module with:
1. Statistical significance testing between models
2. Per-class performance analysis
3. Root cause diagnosis (data quality, preprocessing, model issues)
4. Actionable, prioritized suggestions
5. Learning from past reflections (meta-learning)
"""

from typing import Any, Dict, List, Tuple


def reflect(
        dataset_profile: Dict[str, Any],
        evaluation: Dict[str, Any],
        all_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dataset characteristics
        evaluation: Best model's metrics
        all_metrics: Metrics for all trained models
    
    Returns:
        Dictionary with:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - issues: List[str] (identified problems)
            - suggestions: List[str] (improvement recommendations)
            - replan_recommended: bool (should we replan?)
    
    TODO for students:
    - Implement statistical tests (paired t-tests, Wilcoxon tests)
    - Add per-class performance analysis
    - Detect overfitting vs underfitting
    - Analyze confusion matrix patterns
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """

    best_model = evaluation.get("model")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)

    issues: List[str] = []
    suggestions: List[str] = []

    # Basic comparison with dummy baseline
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)

    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba

        # TODO: Make this more sophisticated
        # Consider: confidence intervals, effect sizes, etc.
        if improvement < 0.05:
            issues.append(
                f"Best model only {improvement:.3f} better than baseline. "
                "Weak signal or pipeline issues."
            )
            suggestions.append(
                "Check for target leakage, verify target quality, "
                "or improve feature engineering."
            )
        elif improvement < 0.10:
            suggestions.append(f"Marginal improvement over baseline: {improvement:.3f} "
                               f"Feature engineering or more expressive models may help")

    # TODO: Add more sophisticated checks

    # Check F1 score
    # TODO: Make threshold adaptive based on problem difficulty
    if imb >= 5.0:
        f1_threshold = 0.45
    elif imb >= 3.0:
        f1_threshold = 0.52
    else:
        f1_threshold = 0.50

    if f1_macro < f1_threshold:
        issues.append(f"Macro F1 {f1_macro:.3f} is below adaptive threshold."
                      f"Set for imbalance ration {imb:.3f}.")
        suggestions.append(
            "Try hyperparameter tuning, ensemble methods "
            "or improve preprocessing."
        )

    # TODO: Add imbalance-specific analysis
    if imb >= 10.0:
        suggestions.append(f"Severe imbalance ratio: {imb:.3f}."
                           f"Apply SMOTE oversampling and set class_weight to balanced.")
    elif imb >= 3.0:
        suggestions.append(
            f"Moderate imbalance ratio: {imb:.3f}."
            f"Consider class_weight, threshold tuning or SMOTE. "
        )

    # TODO: Add checks for:
    # - Model diversity (are all models performing similarly?)
    issues += comp_models_statiscally(all_metrics)

    # - Per-class performance (which classes are problematic?)
    class_issues, class_suggestions = analyse_per_class_performance(evaluation)
    issues += class_issues
    suggestions += class_suggestions

    # - Precision-recall tradeoff
    pr_issues, pr_suggestions = analyse_precison_recall_tradeoff(evaluation)
    issues += pr_issues
    suggestions += pr_suggestions

    # - High-cardinality categorical features

    # - Feature importance patterns

    # - Learning curves (overfitting/underfitting)
    issues += detect_overfitting(evaluation)

    if any("overfitting" in i for i in issues):
        suggestions.append(f"Overfitting detected: add regularisation (C, max_depth)"
                           f"Reduce model complexity or apply cross-validation.")
    if any("underfitting" in i for i in issues):
        suggestions.append(f"Underfitting detected: try more expressive models, add polynomial features"
                           f"Or reduce regularisation strength")

    # Determine status
    status = "needs_attention" if issues else "ok"

    # Simple replanning trigger
    # TODO: Make this more sophisticated
    # replan_recommended = bool(issues and f1_macro < 0.60)
    status = "needs_attention" if issues else "OK"

    trigger_score = len(issues)
    if any("overfitting" in i for i in issues):
        trigger_score += 2
    if any("baseline" in i.lower() for i in issues):
        trigger_score += 1
    if any("similarly" in i for i in issues):
        trigger_score -= 2

    replan_recommended = trigger_score >= 3

    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger replanning based on reflection.
    
    This is a simple policy. Students should implement more sophisticated logic.
    
    TODO for students:
    - Consider multiple factors (performance, confidence, resource budget)
    - Implement diminishing returns detection
    - Use memory to avoid repeating failed strategies
    - Set adaptive thresholds based on problem difficulty
    """
    return bool(reflection.get("replan_recommended", False))


def apply_replan_strategy(
        plan: List[str],
        dataset_profile: Dict[str, Any],
        reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify the plan and dataset profile based on reflection.
    
    This is a very basic implementation. Students should make this sophisticated.
    
    Args:
        plan: Current execution plan
        dataset_profile: Current dataset profile
        reflection: Reflection results
    
    Returns:
        Tuple of (modified_plan, modified_profile)
    
    TODO for students:
    - Implement specific strategies for specific issues
    - Add preprocessing steps based on identified problems
    - Modify model selection based on performance patterns
    - Adjust hyperparameters
    - Try ensemble methods
    - Implement different replan strategies (aggressive, conservative)
    """

    # Copy to avoid modifying originals
    new_plan = list(plan)
    new_profile = dict(dataset_profile)
    issues = reflection.get("issues", [])
    notes = list(new_profile.get("notes", []))

    # Basic strategy: add a note
    # TODO: Implement actual strategy changes
    def insert_before(step: str, new_step: str):
        if new_step in new_plan:
            return
        if step in new_plan:
            new_plan.insert(new_plan.index(step), new_step)
        else:
            new_plan.append(new_step)

    # notes.append("Replan: adjusting strategy after reflection.")
    # new_profile["notes"] = notes
    # new_plan.append("replan_attempt")

    # TODO: Implement sophisticated replan strategies:
    # - If low performance: try ensemble methods
    if any("underfitting" in i for i in issues):
        if "apply_ensemble_methods" not in new_plan:
            new_plan.append("apply_ensemble_methods")
            notes.append("Replan: underfitting detected - adding ensemble methods")

    # - If imbalance issues: add SMOTE or adjust thresholds
    if any("imbalance" in i.lower() for i in issues):
        insert_before("train_models", "apply_smote_oversampling")
        notes.append("Replan: persistent imbalance - escalating to SMOTE oversampling")

    # - If overfitting: add regularization
    if any("overfitting" in i for i in issues):
        insert_before("train_models", "apply_regularisation")
        notes.append("Replan: overfitting detected - adding regularisation step")

    # - If underfitting: increase model complexity
    if any("underfitting" in i for i in issues):
        if "apply_ensemble_methods" not in new_plan:
            new_plan.append("apply_ensemble_methods")
            notes.append("Replan: underfitting detected - adding ensemble methods")

    # - If feature issues: add feature engineering steps
    if any("baseline" in i.lower() for i in issues):
        insert_before("train_models", "apply_feature_engineering")
        notes.append("Replan: weak baseline gap - adding feature engineering step")

    return new_plan, new_profile


# TODO: Add helper functions for reflection
def detect_data_quality_issues(dataset_profile: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    issues, suggestions = [], []
    rows = dataset_profile.get("shape", {}).get("rows", 0)
    cols = dataset_profile.get("shape", {}).get("cols", 0)
    missing_pct = dataset_profile.get("missing_pct", {})
    cat_cols = dataset_profile.get("feature_types", {}).get("categorical", [])
    high_card = dataset_profile.get("high_cardinality_cols", [])

    if rows < 300:
        issues.append(f"Very small dataset ({rows} rows). Model estimated may be unreliable.")
        suggestions.append(f"Use cross-validation instead of a singel train/test split.")
    if cols > rows:
        issues.append(f"More features ({cols} cols) than samples ({rows}) High risk of overfitting.")
        suggestions.append(f"Apply aggressive feature selection or dimensionality reduction before training.")

    fully_missing = [col for col, pct in missing_pct.items() if pct >= 1.0]
    if fully_missing:
        issues.append(f"Columns with 100% missing values detected: {fully_missing}")
        suggestions.append(f"Drop fully missing columns {fully_missing} before preprocessing.")

    if high_card:
        issues.append(f"High cardinality features ({high_card}) detected."
                      f"One-hot encoding will produce very sparse features.")
        suggestions.append(f"Apply target encoding or frequency encoding instead.")

    if len(cat_cols) == cols - 1:
        suggestions.append(f"Dataset almost entirely categorical."
                           f"Ensure encoding strategy handles all columns correctly ")

    return issues, suggestions


def generate_explanation(evaluation: Dict[str, Any], issues: List[str], suggestions: List[str],
                         replan_recommended: bool) -> str:
    best_model = evaluation.get("model", "unknown")
    f1 = float(evaluation.get("f1_macro", 0.0))
    bal_acc = float(evaluation.get("bal_accuracy", 0.0))

    lines = []

    lines.append(f"The best model was: {best_model}, achieving a macro F1 score of {f1:.3f}."
                 f"and a balanced accuracy score of {bal_acc:.3f}.")

    if issues:
        lines.append(f"{len(issues)} issue(s) detected in reflection.")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append(f"No significant issues detected in reflection. Pipeline performing as expected")

    if suggestions:
        lines.append(f"The following improvement(s) are recommended (highest priority first): ")
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")

    if replan_recommended:
        lines.append(f"Based on issued found, replanning is recommended.")
    else:
        lines.append(f"No replanning required at this stage.")
    return "\n".join(lines)


# def compare_models_statistically(...):
def comp_models_statiscally(all_metrics: List[Dict[str, Any]]) -> List[str]:
    issues = []
    scores = [
        (m.get("model"), float(m.get("f1_macro", 0.0)))
        for m in all_metrics
        if "Dummy" not in m.get("model", "")
    ]

    if len(scores) < 2:
        return issues

    best_name, best_score = max(scores, key=lambda x: x[1])
    worst_name, worst_score = min(scores, key=lambda x: x[1])
    spread = best_score - worst_score

    if spread < 0.2:
        issues.append(
            f"Best model only {spread:.3f} better than baseline. "
        )
    elif spread > 0.15:
        issues.append(
            f"Best model only {spread:.3f} better than baseline. "
        )
    return issues


# def detect_overfitting(...):
def detect_overfitting(evaluation: Dict[str, Any]) -> List[str]:
    issues = []
    train_f1 = float(evaluation.get("train_f1_macro", 0.0))
    test_f1 = float(evaluation.get("f1_macro", 0.0))

    if train_f1 <= 0 or test_f1 <= 0:
        return issues

    gap = train_f1 - test_f1

    if gap > 0.15:
        issues.append(f"Best model only {gap:.3f} better than baseline. ")
    elif test_f1 > train_f1 - 0.02 and test_f1 < 0.55:
        issues.append(f"Best model only {gap:.3f} better than baseline. ")
    return issues


# def analyze_per_class_performance(...):
def analyse_per_class_performance(evaluation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    issues, suggestions = [], []
    per_class = evaluation.get("per_class_f1", {})

    if not (per_class):
        return issues, suggestions

    weak_classes = [cls for cls, f1 in per_class.items() if float(f1) < 0.40]
    strong_classes = [cls for cls, f1 in per_class.items() if float(f1) >= 0.80]

    if weak_classes:
        issues.append(f"Poor performance on class(es) {weak_classes} (F1 < 0.40)."
                      f"Likely caused by severe imbalance or label noise.")
        suggestions.append(f"Apply targeted oversampling or cost-sensitive learning for class(es) {weak_classes}.")
    if strong_classes and weak_classes:
        suggestions.append(f"Model performs well on {strong_classes} but struggles on {weak_classes}. "
                           f"Consider a one vs rest strategy on the weak classes")
        return issues, suggestions


def analyse_precison_recall_tradeoff(evaluation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    issues, suggestions = [], []
    precision = float(evaluation.get("precision_macro", 0.0))
    recall = float(evaluation.get("recall_macro", 0.0))

    if precision <= 0 or recall <= 0:
        return issues, suggestions

    gap = abs(recall - precision)

    if gap > 0.15:
        if precision > recall:
            issues.append(f"High precision {precision:.3f} but low recall {recall:.3f}. "
                          f"Model is too conservative.")
            suggestions.append(
                f"Lower the classification threshold or increase minority class weight to improve recall.")
        else:
            issues.append(f"High recall {recall:.3f} but low precision {precision:.3f}. "
                          f"Model is over-predicting positives")
            suggestions.append(
                f"Raise the classification threshold or add stringer regularisation to improve precision.")
    return issues, suggestions


# def prioritize_suggestions(...):
def prioritise_suggestions(suggestions: List[str]) -> List[str]:
    priority_keywords = [
        "SMOTE", "class_weight", "threshold", "feature engineering", "ensemble", "oversampling", "targeted"
    ]
    high = [s for s in suggestions if any(k.lower() for k in priority_keywords)]
    low = [s for s in suggestions if s not in high]
    return high + low
