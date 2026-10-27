import json
import os
import shutil
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    #return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def similarity_score(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> float:
    score = 0.0
    comparisons = 0

    def normalised_closeness(a, b, scale) -> float:
        if a is None or b is None:
            return 0.0
        if a == 0 and b == 0:
            return 1.0
        return max(0.0, 1.0 - abs(a - b) / scale)

    score += normalised_closeness(profile_a.get("rows"), profile_b.get("rows"), 50000)
    comparisons += 1

    score += normalised_closeness(profile_a.get("cols"), profile_b.get("cols"), 500)
    comparisons += 1

    score += normalised_closeness(profile_a.get("imbalance_ratio"), profile_b.get("imbalance_ratio"), 20.0)
    comparisons += 1

    score += normalised_closeness(profile_a.get("missing_pct_max"), profile_b.get("missing_pct_max"), 1.0)
    comparisons += 1

    return score / comparisons if comparisons > 0 else 0.0


class JSONMemory:
    """
    Lightweight persistent memory for the agent.
    Stores dataset fingerprint -> best model/metrics and notes.
    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {"datasets": {}, "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}]}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_dataset_record(self, fingerprint: str) -> Optional[dict[str, Any]]:
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(self, fingerprint: str, record: dict[str, Any]) -> None:
        self.data.setdefault("datasets", {})[fingerprint] = record
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    def store_experience(self, fingerprint: str, dataset_profile: dict[str, Any], plan: List[str],
                         metrics: dict[str, Any],
                         reflection: dict[str, Any], replan_triggered: bool = False,
                         failed_steps: Optional[List[str]] = None,
                         best_model=None) -> None:
        record = {
            "fingerprint": fingerprint,
            "timestamp": now_iso(),
            "dataset_summary": {
                "rows": dataset_profile.get("shape", {}).get("rows"),
                "cols": dataset_profile.get("shape", {}).get("columns"),
                "imbalance_ratio": dataset_profile.get("imbalance_ratio"),
                "missing_pct_max": max(dataset_profile.get("missing_pct", {}).values(), default=0),
                "n_numeric": len(dataset_profile.get("feature_types", {}).get("numeric", [])),
                "n_categorical": len(dataset_profile.get("feature_types", {}).get("categorical", [])),
            },
            "plan_used": plan,
            "best_model": best_model,
            "metrics": metrics,
            "reflection_status": reflection.get("status"),
            "issues_found": reflection.get("issues", []),
            "replan_triggered": replan_triggered,
            "failed_steps": failed_steps or [],

        }
        self.data.setdefault("datasets", {})[fingerprint] = record
        self.save()

    def find_similar_experience(self, dataset_profile: dict[str, Any], top_n: int = 1, threshold: float = 0.6,
                                dataset_summary=None) -> List[Tuple[float, dict[str, Any]]]:
        current_summary = {
            "rows": dataset_profile.get("shape", {}).get("rows"),
            "cols": dataset_profile.get("shape", {}).get("columns"),
            "imbalance_ratio": dataset_profile.get("imbalance_ratio"),
            "missing_pct_max": max(dataset_profile.get("missing_pct", {}).values(), default=0),
        }

        results = []
        for fingerprint, record in self.data.get("datasets", {}).items():
            past_summary = record.get("dataset_summary", {})
            score = similarity_score(current_summary, past_summary)
            if score >= threshold:
                results.append((score, record))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    def get_hint_for_profile(self, dataset_profile: dict[str, Any]) -> Optional[dict[str, Any]]:
        matches = self.find_similar_experience(dataset_profile, top_n=1, threshold=0.6)

        if not matches:
            return None

        score, record = matches[0]
        return {
            "best_model": record.get("best_model"),
            "failed_steps": record.get("failed_steps", []),
            "replan_triggered": record.get("replan_triggered", False),
            "similarity_score": score,
            "source_fingerprint": record.get("fingerprint"),
        }

    def get_best_model_for_profile(self, dataset_profile: dict[str, Any]) -> Optional[str]:
        hint = self.get_hint_for_profile(dataset_profile)
        return hint.get("best_model") if hint else None

    def failure_summary(self) -> dict[str, int]:
        failure_counts: dict[str, int] = {}
        for record in self.data.get("datasets", {}).values():
            for step in record.get("failed_steps", []):
                failure_counts[step] = failure_counts.get(step, 0) + 1
        return dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True))

    def get_model_win_counts(self) -> Dict[str, int]:
        win_counts: dict[str, int] = {}
        for record in self.data.get("datasets", {}).values():
            model = record.get("best_model")
            if model:
                win_counts[model] = win_counts.get(model, 0) + 1
        return dict(sorted(win_counts.items(), key=lambda x: x[1], reverse=True))

    def summarise(self) -> dict[str, Any]:
        datasets = self.data.get("datasets", {})
        return {
            "total_runs": len(datasets),
            "model_win_counts": self.get_model_win_counts(),
            "step_failure_counts": self.failure_summary(),
            "replan_rate": (
                sum(1 for r in datasets.values() if r.get("replan_triggered")) / len(datasets) if datasets else 0.0
            )
        }

