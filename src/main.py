"""Main entry point for the Medical Diagnosis Simulation System."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.helper import (
    build_arg_parser,
    build_final_output,
    build_progress_output,
    compute_dataset_accuracies,
    compute_result_counts,
    compute_time_stats,
    get_case_dataset,
    print_summary,
    save_results,
)
from src.models.patient_case import PatientCaseFile
from src.models.diagnosis import CaseResult, EvaluationSummary
from src.workflows.workflow_engine import WorkflowEngine


async def run_evaluation(
    input_file: str,
    workflow_config: str,
    output_file: str | None = None,
    verbose: bool = False,
    case_range_override: str | None = None,
    resume: bool = False,
    datasets: list[str] | None = None,
    max_cases: int | None = None,
    case_ids: list[str] | None = None
) -> EvaluationSummary:
    """Run the evaluation on all patient cases.

    Args:
        input_file: Path to JSON file with patient cases
        workflow_config: Path to workflow YAML config
        output_file: Optional path to save results JSON
        verbose: Whether to print detailed progress
        case_range_override: CLI override for case range (e.g., "0-20", "20-40")
        resume: If True and ``output_file`` exists, skip cases already marked
            complete and append new results to the same file
        datasets: If provided (and not ``["all"]``), only run cases whose
            ``case_id`` prefix matches one of these dataset names
        max_cases: If set, run at most this many cases (applied after
            ``--resume`` filtering) then stop
        case_ids: If provided, restrict the run to this explicit list of case
            IDs (takes precedence over ``datasets`` and ``case_range_override``)

    Returns:
        EvaluationSummary with all results
    """
    # Load patient cases
    print(f"Loading patient cases from: {input_file}")
    case_file = PatientCaseFile.from_json_file(input_file)
    print(f"Loaded {len(case_file.cases)} cases")
    
    # Initialize workflow engine
    print(f"Loading workflow config from: {workflow_config}")
    engine = WorkflowEngine(config_path=workflow_config)
    print(f"Workflow: {engine.config.get('name', 'unknown')}")
    print(f"Description: {engine.config.get('description', '')}")
    if engine.config.get("unstructured_mode", False):
        print("Running in unstructured output mode (LLM generates text, then we parse/validate JSON)")
    else:
        print("Running in structured output mode (LLM generates native JSON with strict format)")
    # Get case range from CLI override or config (supports: "10-20", "10:20", 10, or empty for all)
    case_range = case_range_override or engine.config.get('case_range') or engine.config.get('num_cases')
    cases_to_process = case_file.cases

    # Filter to a specific list of case IDs if provided (takes precedence over
    # dataset/range filtering — useful for qualitative reruns of single cases).
    if case_ids:
        wanted = set(case_ids)
        available_ids = {c.case_id for c in cases_to_process}
        missing = sorted(wanted - available_ids)
        cases_to_process = [c for c in cases_to_process if c.case_id in wanted]
        print(f"\nFiltered to {len(cases_to_process)} case(s) by --case-ids: "
              f"{', '.join(c.case_id for c in cases_to_process)}")
        if missing:
            print(f"  [WARN] Requested but not found in input file: {missing}")
        if not cases_to_process:
            print("  No matching cases — nothing to run.")
            return EvaluationSummary(
                total_cases=0, correct=0, incorrect=0,
                accuracy=0.0, avg_confidence=0.0, avg_rounds=0.0, results=[]
            )

    # Filter by datasets if specified (skipped when case_ids was given)
    if not case_ids and datasets and datasets != ["all"]:
        cases_to_process = [c for c in cases_to_process if get_case_dataset(c.case_id) in datasets]
        print(f"Filtered to datasets: {', '.join(datasets)} → {len(cases_to_process)} cases")
    
    if case_range:
        if isinstance(case_range, str) and ('-' in case_range or ':' in case_range):
            # Parse range format: "10-20" or "10:20"
            separator = '-' if '-' in case_range else ':'
            start, end = case_range.split(separator)
            start_idx = int(start.strip()) if start.strip() else 0
            end_idx = int(end.strip()) if end.strip() else len(cases_to_process)
            cases_to_process = cases_to_process[start_idx:end_idx]
            print(f"Processing cases {start_idx} to {end_idx} ({len(cases_to_process)} cases)")
        elif isinstance(case_range, int) and case_range > 0:
            # Legacy: single integer means first N cases
            cases_to_process = cases_to_process[:case_range]
            print(f"Processing first {case_range} of {len(cases_to_process)} cases (limited by config)")
        else:
            print(f"Processing all {len(cases_to_process)} cases")
    else:
        print(f"Processing all {len(cases_to_process)} cases")
    print("-" * 60)
    
    # Resume: load existing results and skip completed cases
    results: list[CaseResult] = []
    completed_case_ids: set[str] = set()
    
    if resume and output_file and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", [])
        for r in existing_results:
            cid = r.get("case_id", "")
            status = r.get("completion_status", "complete")
            result_val = r.get("result", "")
            # Skip incomplete/error cases — they should be re-run
            if status in ("incomplete", "error") or result_val == "incomplete":
                continue
            completed_case_ids.add(cid)
            results.append(CaseResult(**r))
        print(f"Resume: loaded {len(completed_case_ids)} completed cases from {output_file}")
        cases_to_process = [c for c in cases_to_process if c.case_id not in completed_case_ids]
        print(f"Resume: {len(cases_to_process)} cases remaining to process")
        print("-" * 60)
    
    # Apply max_cases limit (process at most N cases then stop)
    original_remaining = len(cases_to_process)
    if max_cases and max_cases > 0 and len(cases_to_process) > max_cases:
        print(f"Max cases: limiting to {max_cases} of {len(cases_to_process)} remaining cases")
        cases_to_process = cases_to_process[:max_cases]
    
    # total_cases reflects the full scope (completed + ALL remaining), not just this batch
    total_cases = original_remaining + len(completed_case_ids)
    
    for i, case in enumerate(cases_to_process, len(completed_case_ids) + 1):
        print(f"\n[Case {i}/{total_cases}] {case.case_id}")
        # print("Case patient history:\n")
        # print(case.patient_history)
        # print("\nDone")
        try:
            result = await engine.run(
                patient_history=case.patient_history,
                ground_truth=case.gt,
                case_id=case.case_id
            )
            results.append(result)
            
            # Print result
            status = "✓" if result.result == "correct" else "✗"
            print(f"  {status} Predicted: {result.predicted_disease}")
            print(f"    Ground Truth: {result.ground_truth}")
            print(f"    Confidence: {result.confidence}%")
            print(f"    Rounds: {result.rounds}")
            
            if verbose and result.dialog_history:
                print(f"\n  Dialog:\n{result.dialog_history}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append(CaseResult(
                case_id=case.case_id,
                ground_truth=case.gt,
                predicted_disease="Error",
                confidence=0,
                result="incomplete",
                rounds=0,
                completion_status="error",
                error_log=[str(e)]
            ))
        
        # Save results to JSON file after each case
        if output_file:
            save_results(
                output_file,
                build_progress_output(
                    workflow_name=engine.config.get("name", "unknown"),
                    input_file=input_file,
                    results=results,
                    completed=i,
                    total_cases=total_cases,
                ),
            )
    
    # Calculate summary statistics
    correct_count, incorrect_count, incomplete_count = compute_result_counts(results)
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
    avg_rounds = sum(r.rounds for r in results) / len(results) if results else 0
    time_stats = compute_time_stats(results)
    
    summary = EvaluationSummary(
        total_cases=len(results),
        correct=correct_count,
        incorrect=incorrect_count,
        accuracy=correct_count / len(results) * 100 if results else 0,
        avg_confidence=avg_confidence,
        avg_rounds=avg_rounds,
        results=results
    )
    
    print_summary(summary, incomplete_count, time_stats)
    
    # Save results if output file specified
    if output_file:
        save_results(
            output_file,
            build_final_output(
                workflow_name=engine.config.get("name", "unknown"),
                input_file=input_file,
                summary=summary,
                results=results,
                total_cases=total_cases,
                incomplete_count=incomplete_count,
            ),
        )
        print(f"\nResults saved to: {output_file}")
    
    return summary


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Apply unstructured mode to settings if flag is set
    if args.unstructured:
        from src.config.settings import Settings
        # Override the cached settings
        import src.config.settings as settings_module
        settings_module.get_settings.cache_clear()
        os.environ['UNSTRUCTURED_MODE'] = 'true'
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Validate workflow config exists
    if not os.path.exists(args.workflow):
        print(f"Error: Workflow config not found: {args.workflow}")
        sys.exit(1)
    
    # Create output directory if needed
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    try:
        asyncio.run(run_evaluation(
            input_file=args.input,
            workflow_config=args.workflow,
            output_file=args.output,
            verbose=args.verbose,
            case_range_override=getattr(args, 'case_range', None),
            resume=getattr(args, 'resume', False),
            datasets=getattr(args, 'datasets', None),
            max_cases=getattr(args, 'max_cases', None),
            case_ids=getattr(args, 'case_ids', None)
        ))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
