"""Pure helpers for src/main.py.

Keeps main.py focused on orchestration (run_evaluation / CLI dispatch).
Everything here is data manipulation, I/O building, stats, or CLI plumbing
— no workflow logic.
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime

from src.models.diagnosis import CaseResult, EvaluationSummary


# Known dataset name prefixes for case_id parsing. These are the prefixes
# emitted by the loader scripts in ``scripts/`` and the cleaned MeDxBench
# input file. Order does not matter — the longest matching prefix wins.
_KNOWN_DATASETS = {
    "medreason_medqa", "diagnosis_arena", "medreason_medmcqa",
    "craftmd_derma", "medmcqa", "pubmed",
}


def get_case_dataset(case_id: str) -> str:
    """Return the dataset name that ``case_id`` belongs to.

    Uses longest-prefix matching against ``_KNOWN_DATASETS``; falls back to
    the first two underscore-separated parts when no known prefix matches.
    """
    parts = case_id.split("_")
    for prefix_len in range(len(parts), 0, -1):
        candidate = "_".join(parts[:prefix_len])
        if candidate in _KNOWN_DATASETS:
            return candidate
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def compute_dataset_accuracies(results: list) -> dict:
    """Compute per-dataset accuracy from results (accepts CaseResult or dict)."""
    dataset_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        case_id = r.case_id if isinstance(r, CaseResult) else r.get("case_id", "")
        result_val = r.result if isinstance(r, CaseResult) else r.get("result", "")
        dataset_name = get_case_dataset(case_id)
        dataset_stats[dataset_name]["total"] += 1
        if result_val == "correct":
            dataset_stats[dataset_name]["correct"] += 1
    return {
        name: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
        }
        for name, s in sorted(dataset_stats.items())
    }


def _get_field(r, name: str):
    """Read a field from either a CaseResult instance or a dict."""
    return getattr(r, name) if isinstance(r, CaseResult) else r.get(name)


def _is_incomplete(r) -> bool:
    """Whether a CaseResult or dict represents an incomplete/errored case."""
    return _get_field(r, "result") == "incomplete" or _get_field(r, "completion_status") in ("incomplete", "error")


def compute_result_counts(results: list) -> tuple[int, int, int]:
    """Return ``(correct, incorrect, incomplete)`` counts (accepts CaseResult or dict)."""
    correct = sum(1 for r in results if _get_field(r, "result") == "correct")
    incorrect = sum(1 for r in results if _get_field(r, "result") == "incorrect")
    incomplete = sum(1 for r in results if _is_incomplete(r))
    return correct, incorrect, incomplete


def compute_time_stats(results: list[CaseResult]) -> dict:
    """Return time-per-case statistics rounded to 2 decimals."""
    exec_times = [r.execution_time_seconds for r in results if r.execution_time_seconds is not None]
    if not exec_times:
        return {"mean": 0, "stdev": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(exec_times), 2),
        "stdev": round(statistics.stdev(exec_times), 2) if len(exec_times) > 1 else 0,
        "min": round(min(exec_times), 2),
        "max": round(max(exec_times), 2),
    }


def build_progress_output(
    workflow_name: str,
    input_file: str,
    results: list[CaseResult],
    completed: int,
    total_cases: int,
) -> dict:
    """Output dict written to the results file after each case during a run."""
    correct, incorrect, incomplete = compute_result_counts(results)
    return {
        "timestamp": datetime.now().isoformat(),
        "workflow": workflow_name,
        "input_file": input_file,
        "progress": {"completed": completed, "total": total_cases},
        "summary": {
            "cases_processed": len(results),
            "correct": correct,
            "incorrect": incorrect,
            "INCOMPLETE": incomplete,
            "accuracy": correct / len(results) * 100 if results else 0,
            "time_per_case": compute_time_stats(results),
        },
        "dataset_accuracies": compute_dataset_accuracies(results),
        "results": [r.model_dump() for r in results],
    }


def build_final_output(
    workflow_name: str,
    input_file: str,
    summary: EvaluationSummary,
    results: list[CaseResult],
    total_cases: int,
    incomplete_count: int,
) -> dict:
    """Output dict written to the results file at the end of a run."""
    return {
        "timestamp": datetime.now().isoformat(),
        "workflow": workflow_name,
        "input_file": input_file,
        "progress": {"completed": len(results), "total": total_cases},
        "summary": {
            "total_cases": summary.total_cases,
            "correct": summary.correct,
            "incorrect": summary.incorrect,
            "INCOMPLETE": incomplete_count,
            "accuracy": summary.accuracy,
            "avg_confidence": summary.avg_confidence,
            "avg_rounds": summary.avg_rounds,
            "time_per_case": compute_time_stats(results),
        },
        "dataset_accuracies": compute_dataset_accuracies(results),
        "results": [r.model_dump() for r in results],
    }


def save_results(output_file: str, output_data: dict) -> None:
    """Write the output dict to disk as JSON."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def print_summary(summary: EvaluationSummary, incomplete_count: int, time_stats: dict) -> None:
    """Print the evaluation summary block."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Cases: {summary.total_cases}")
    print(f"Correct: {summary.correct}")
    print(f"Incorrect: {summary.incorrect}")
    if incomplete_count > 0:
        print(f"INCOMPLETE: {incomplete_count}")
    print(f"Accuracy: {summary.accuracy:.1f}%")
    print(f"Average Confidence: {summary.avg_confidence:.1f}%")
    print(f"Average Rounds: {summary.avg_rounds:.1f}")
    print(
        f"Time per Case: {time_stats['mean']:.2f} ± {time_stats['stdev']:.2f} sec "
        f"(min: {time_stats['min']:.2f}, max: {time_stats['max']:.2f})"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for src/main.py."""
    parser = argparse.ArgumentParser(
        description="Medical Diagnosis Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full MeDxAgent on all MeDxBench datasets, saving results
  python -m src.main -i data/input/medxbench.json -w configs/variations/medxagent.yaml -o data/output/results.json --datasets all

  # Run a small range of cases for a quick check
  python -m src.main -i data/input/medxbench.json -w configs/variations/medxagent.yaml -r 0-20 -o data/output/results.json

  # Resume an interrupted run (skips cases already completed in the output file)
  python -m src.main -i data/input/medxbench.json -w configs/variations/medxagent.yaml -o data/output/results.json --resume
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="Path to JSON file with patient cases")
    parser.add_argument("-w", "--workflow", required=True, help="Path to workflow YAML config file")
    parser.add_argument("-o", "--output", help="Path to save results JSON (optional)")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress including dialog history",
    )
    parser.add_argument(
        "-r", "--case-range",
        help="Case range to process (e.g., '0-20', '20-40'). Overrides config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file: skip already-completed cases and re-run any incomplete/error cases plus the remaining ones (requires -o)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Process at most N cases then stop. Works with --resume to process a subset of remaining cases.",
    )
    parser.add_argument(
        "--unstructured",
        action="store_true",
        help="Use unstructured output mode (LLM generates text, then validate JSON with retry). Default uses native structured output parsing.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["craftmd_derma", "diagnosis_arena", "medmcqa", "medreason_medqa", "pubmed"],
        help="Only process cases from these datasets. Use 'all' for all datasets. Default: 5 selected datasets.",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        nargs="+",
        default=None,
        help="Only run these specific case IDs (e.g. --case-ids pubmed_pbm_35800852 medmcqa_abc). "
             "Takes precedence over --datasets and --case-range.",
    )
    return parser
