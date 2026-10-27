# Orchestrator for an "agentic" offline data scientist pipeline.
# Handles dataset loading, profiling, planning, training, evaluation, reflection,
# and optional re-planning cycles. Designed primarily for classification tasks.
import os
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

# Agent components and tooling used by the orchestrator
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models
from tools.evaluation import evaluate_best, write_markdown_report, save_json


# Lightweight container for run metadata and parameters
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    #return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def estimate_plan_cost(plan):  # For cost-aware planning
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


class AgenticDataScientist:
    """
    Offline Agentic Data Scientist (classification-focused).

    Responsibilities:
    - load and profile datasets
    - create a plan (via planner)
    - build preprocessors and select candidate models
    - train and evaluate models
    - reflect on results and optionally re-plan
    - persist artefacts and update memory
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        # Verbose controls logging output
        self.verbose = verbose
        # Simple persistent memory used to remember prior runs for a dataset fingerprint
        self.memory = JSONMemory(memory_path)

        # Context and transient state populated when run() is executed
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        """Print a log message when verbose is enabled."""
        if self.verbose:
            print(f"[{now_iso()}] [AgenticDataScientist] {msg}")

    def log_section(self, title: str) -> None:
        """Print a section divider for readability in long runs."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f" {title}")
            print(f"{'=' * 60}")

    def load_data(self, path: str) -> pd.DataFrame:
        """Load a CSV into a pandas DataFrame and log its shape."""
        self.log(f"Loading dataset: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {path}: {e}")
        if df.empty:
            raise ValueError(f"Dataset at {path} not found. ")

        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    def execute_with_retry(
            self,
            step_name: str,
            fn,
            max_retry: int = 3,
            failed_steps: List[str] = None,
    ) -> Any:

        for attempt in range(1, max_retry + 1):
            try:
                self.log(f"Step {step_name} - attempt {attempt}/{max_retry}")
                result = fn()
                self.log(f"Step {step_name} - completed successfully")
                return result
            except Exception as e:
                self.log(f"Step {step_name} - attempt {attempt} failed: {e}")
                if attempt == max_retry:
                    self.log(f"Step {step_name} - failed to complete successfully")
                    if failed_steps is not None:
                        failed_steps.append(step_name)
        return None

    def should_execute_step(self, step: str, plan: List[str]) -> bool:
        skip_prefixes = ("prioritise_model:", "fallback_for:", "fallback_strategy:")
        return step in plan and not any(step.startswith(p) for p in skip_prefixes)

    def run(
            self,
            data_path: str,
            target: str,
            output_root: str = "outputs",
            seed: int = 42,
            test_size: float = 0.2,
            max_replans: int = 1,
    ) -> str:
        """
        Main orchestration entry point.

        Parameters:
        - data_path: path to the CSV dataset
        - target: target column name or 'auto' to infer
        - output_root: directory where outputs are stored (subdir will be created)
        - seed/test_size: training reproducibility and test split
        - max_replans: maximum number of times to re-plan and re-run

        Returns: path to the output directory for this run
        """
        self.log_section("Starting Run")

        # Create a unique run id and output directory for artefacts
        run_id = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        #run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Populate run context with parameters and metadata
        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        # Internal state used to track replanning attempts
        self.state = {"replan_count": 0, "failed_steps": []}

        # Load dataset into memory
        self.log_section("Loading Dataset")
        df = self.load_data(data_path)

        # If client requested auto target detection, infer it from data
        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError("Could not infer target column. Please provide --target <name>.")
            # Update context with inferred target name
            self.ctx.target = inferred
            self.log(f"Inferred target: {inferred}")

        # Produce a dataset profile (EDA summary) and a fingerprint used for memory
        self.log_section("Profiling Dataset")
        profile = profile_dataset(df, self.ctx.target)
        fp = dataset_fingerprint(df, self.ctx.target)
        self.log(f"Dataset fingerprint: {fp}")
        self.log(
            f"Profile summary - rows: {profile['shape']['rows']},"
            f"cols: {profile['shape']['cols']},"
            f"imbalance: {profile.get('imbalance_ratio', 'N/A')}"
            f"numeric features: {len(profile.get('feature_types', {}).get('numeric', []))}"
            f"categorical features: {len(profile.get('feature_types', {}).get('categorical', []))}"
        )

        # Look up previous runs for the same dataset fingerprint (memory hint)
        self.log_section("Checking Memory")
        prev = self.memory.get_dataset_record(fp)
        if prev:
            self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fp}")
        else:
            hint = self.memory.get_hint_for_profile(profile)
            if hint:
                self.log(
                    f"Similar dataset found in memory - "
                    f"similarity = {hint['similarity_score']:.2f}, "
                    f"best_model = {hint.get('best_model')},"
                    f"source = {hint.get('source_fingerprint')}"
                )
                prev = hint
            else:
                self.log("No similar past experience found in memory")

        # Create an initial plan informed by the profile and optional memory hint
        self.log_section("Planning")
        plan = create_plan(profile, memory_hint=prev)
        self.log(f"Initial plan: {plan}")
        self.log(f"Estimated plan cost: {estimate_plan_cost(plan)}")

        # Execution loop: trains and evaluates, then optionally replans and repeats
        replan_triggered = False

        while True:
            self.log_section(
                f"Execution (replan #{self.state['replan_count']})"
                if self.state["replan_count"] > 0
                else "Execution"
            )

            # Build preprocessing pipeline tailored to the profile
            preprocessor = self.execute_with_retry(
                "build preprocessor",
                lambda: build_preprocessor(profile),
                failed_steps=self.state["failed_steps"]
            )
            if preprocessor is None:
                self.log("Preprocessor failed. Aborting run.")
                break
            # Choose candidate models to try based on the profile
            candidates = self.execute_with_retry(
                "select_models",
                lambda: select_models(profile, seed=self.ctx.seed),
                failed_steps=self.state["failed_steps"]
            )
            if candidates is None:
                self.log("Model selection failed. Aborting run.")
                break

            self.log(f"Candidate models: {[n for n, _ in candidates]}")

            prioritised = next(
                (s.split(":")[1] for s in plan if s.startswith("prioritise_model:")),
                None,
            )
            if prioritised:
                self.log(f"Memory hint: prioritising model '{prioritised}'")

            # Train candidate models and persist intermediate artefacts
            results = self.execute_with_retry(
                "train models",
                lambda: train_models(
                    df=df,
                    target=self.ctx.target,
                    preprocessor=preprocessor,
                    candidates=candidates,
                    seed=self.ctx.seed,
                    test_size=self.ctx.test_size,
                    output_dir=self.ctx.output_dir,
                    verbose=self.verbose,
                ),
                failed_steps=self.state["failed_steps"]
            )
            if not results:
                self.log("Training failed. Aborting run.")
                break

            # Evaluate the trained models and pick the best one
            eval_payload = self.execute_with_retry(
                "evaluate",
                lambda: evaluate_best(results, output_dir=self.ctx.output_dir),
                failed_steps=self.state["failed_steps"])
            if eval_payload is None:
                self.log("Evaluation failed. Aborting run.")
                break

            best_model_name = eval_payload["best_metrics"]["model"]
            best_f1 = eval_payload["best_metrics"].get("f1_macro", "N/A")
            self.log(f"Best model: {best_model_name} | F1 macro: {best_f1}")

            # Reflect on the evaluation in the context of the dataset profile
            self.log_section("Reflection")
            reflection = reflect(
                dataset_profile=profile,
                evaluation=eval_payload["best_metrics"],
                all_metrics=eval_payload["all_metrics"],
            )
            self.log(f"Reflection status: {reflection['status']}")
            if reflection["issues"]:
                for issue in reflection["issues"]:
                    self.log(f" Issue: {issue}")
            if reflection["suggestions"]:
                for suggestion in reflection["suggestions"]:
                    self.log(f" Suggestion: {suggestion}")

            # Persist core run artefacts for later review
            self.log_section("Saving Artefacts")
            save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
            save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": plan})
            save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
            save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

            # Generate a human-readable markdown report summarising the run
            write_markdown_report(
                out_path=os.path.join(self.ctx.output_dir, "report.md"),
                ctx=self.ctx,
                fingerprint=fp,
                dataset_profile=profile,
                plan=plan,
                eval_payload=eval_payload,
                reflection=reflection,
            )
            self.log(f"Artefacts saved to: {self.ctx.output_dir}")

            # Update the memory store with outcomes from this run
            self.memory.upsert_dataset_record(fp, {
                "last_seen": now_iso(),
                "target": self.ctx.target,
                "shape": profile["shape"],
                "best_model": eval_payload["best_metrics"]["model"],
                "best_metrics": eval_payload["best_metrics"],
            })
            self.log("Memory updated with this run's experience.")

            # Decide whether the agent should attempt to re-plan and re-run
            if not should_replan(reflection):
                # No replan suggested — finish the run
                self.log("No replanning recommended. Run completed.")
                break

            # If we've already replanned the allowed number of times, stop
            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log(f"Replan suggested, but max_replans = {self.ctx.max_replans} reached. Stopping."
                         f"Updated plan: {plan}")
                break

            # Otherwise, increment replan counter and apply the replan strategy
            self.state["replan_count"] += 1
            replan_triggered = True
            self.log(f"Replanning attempt #{self.state['replan_count']}...")

            # apply_replan_strategy returns an updated (plan, profile) pair
            plan, profile = apply_replan_strategy(plan, profile, reflection)
            self.log("Updated plan: {plan}")

        # Final log and return the directory containing run outputs
        self.log_section("Run Complete")
        self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
        self.log(f"Total replan attempts: {self.state['replan_count']}")
        if self.state["failed_steps"]:
            self.log(f"Failed steps: {self.state['failed_steps']}")
        self.log(f"Memory Summary: {self.memory.summarise()}")

        return self.ctx.output_dir

