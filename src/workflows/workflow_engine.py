"""Workflow engine for orchestrating agent execution."""

import asyncio
import json
import time
from typing import Any
import yaml

from src.agents import get_agent, AGENT_REGISTRY
from src.agents.base_agent import BaseAgent
from src.llm import LLMClient, create_llm_client
from src.config import get_settings
from src.models.conversation import Dialog
from src.models.diagnosis import DiagnosisResult, CaseResult
from src.workflows.workflow_context import WorkflowContext


class WorkflowEngine:
    """Engine for executing configured workflows.
    
    The workflow engine:
    - Loads workflow configuration from YAML
    - Orchestrates agent execution based on the workflow
    - Manages dialog state and exit conditions
    - Implements verification loops
    - Supports multiple workflow variants
    """
    
    def __init__(
        self,
        config_path: str | None = None,
        config: dict | None = None,
        llm_client: LLMClient | None = None
    ):
        """Initialize the workflow engine.
        
        Args:
            config_path: Path to workflow YAML config file
            config: Direct config dictionary (alternative to file)
            llm_client: LLM client to use (creates default if not provided)
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.llm = llm_client or create_llm_client()
        self.settings = get_settings()
        self._agents: dict[str, BaseAgent] = {}
        self._initialize_agents()
    
    def _load_config(self, config_path: str) -> dict:
        """Load workflow configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _initialize_agents(self) -> None:
        """Initialize agents specified in the workflow config.
        
        Supports per-agent model overrides via 'agent_models' config:
            agent_models:
              doctor_q_agent:
                model: ...
        """
        agents_config = self.config.get("agents", [])
        agent_models = self.config.get("agent_models", {})
        
        # Cache LLM clients by (model, instance, api_version) to reuse
        llm_cache: dict[tuple, LLMClient] = {}
        
        for agent_name in agents_config:
            if agent_name in AGENT_REGISTRY:
                agent_class = get_agent(agent_name)
                
                # Check if this agent has a model override
                model_override = agent_models.get(agent_name)
                if model_override:
                    print("USING MODEL OVERRIDE for agent", agent_name, model_override)
                    provider = model_override.get("provider", "openai")
                    cache_key = (
                        provider,
                        model_override.get("model", ""),
                        model_override.get("endpoint", ""),
                    )
                    if cache_key not in llm_cache:
                        model_name = model_override["model"]
                        factory_kwargs = {
                            "provider": provider,
                            "model": model_name,
                            "temperature": model_override.get("temperature", 0),
                        }
                        llm_cache[cache_key] = create_llm_client(**factory_kwargs)
                    agent_llm = llm_cache[cache_key]
                    self._agents[agent_name] = agent_class(agent_llm, config=self.config)
                else:
                    self._agents[agent_name] = agent_class(self.llm, config=self.config)
    
    def get_agent(self, name: str) -> BaseAgent:
        """Get an initialized agent by name."""
        if name not in self._agents:
            # Try to initialize on demand
            if name in AGENT_REGISTRY:
                agent_class = get_agent(name)
                # On-demand agents use default LLM (no model override)
                self._agents[name] = agent_class(self.llm, config=self.config)
            else:
                raise ValueError(f"Unknown agent: {name}")
        return self._agents[name]
    
    def _is_idk_answer(self, answer: str) -> bool:
        """Check if the patient's answer indicates they don't know.
        
        When a patient says 'I don't know', we should skip diagnosis agents
        and have the doctor ask the next question without counting this as a round.
        """
        normalized = answer.strip().lower().rstrip('.')
        idk_phrases = [
            "i don't know",
            "i do not know",
            "i dont know",
            "don't know",
            "do not know",
            "not sure",
            "i'm not sure",
            "i am not sure",
            "unknown",
            "no idea",
            "i have no idea"
        ]
        return normalized in idk_phrases or any(normalized == phrase for phrase in idk_phrases)
    
    def _print_token_summary(self):
        """Print per-agent and total token usage summary for the case."""
        parts = []
        total_in, total_out = 0, 0
        for name, agent in self._agents.items():
            a_in, a_out = agent.get_total_tokens()
            if a_in > 0 or a_out > 0:
                parts.append(f"{name}(in={a_in:,},out={a_out:,})")
                total_in += a_in
                total_out += a_out
        if total_in > 0 or total_out > 0:
            breakdown = " | ".join(parts)
            # print(f"  [TOKENS] Total — Input: {total_in:,} | Output: {total_out:,} | Total: {total_in + total_out:,}")
            # print(f"  [TOKENS] Per-Agent Total — {breakdown}")
    
    def _print_turn_tokens(self, round_num: int | None = None):
        """Print per-agent and total token usage for the current turn."""
        parts = []
        turn_in, turn_out = 0, 0
        for name, agent in self._agents.items():
            a_in, a_out = agent.get_turn_tokens()
            if a_in > 0 or a_out > 0:
                parts.append(f"{name}(in={a_in:,},out={a_out:,})")
                turn_in += a_in
                turn_out += a_out
        if turn_in > 0 or turn_out > 0:
            breakdown = " | ".join(parts)
            label = f"Turn {round_num}" if round_num is not None else "Turn"
            # print(f"  [TOKENS] {label} — Input: {turn_in:,} | Output: {turn_out:,} | Total: {turn_in + turn_out:,}")
            # print(f"  [TOKENS] Per-Agent {label} — {breakdown}")
    
    
    async def run(
        self,
        patient_history: str,
        ground_truth: str,
        case_id: str = "unknown"
    ) -> CaseResult:
        """Run the workflow for a single patient case.
        
        Args:
            patient_history: The patient's medical history
            ground_truth: The ground truth disease diagnosis
            case_id: Identifier for the case
            
        Returns:
            CaseResult with the evaluation outcome
        """
        context = WorkflowContext()
        context.set("patient_history", patient_history)
        context.set("ground_truth", ground_truth)
        context.set("case_id", case_id)
        
        # Reset per-agent total token counters for this case
        for agent in self._agents.values():
            agent.reset_total_tokens()
            agent.reset_turn_tokens()
        
        workflow_type = self.config.get("workflow_type", "basic")
        
        # Start timing
        start_time = time.time()
        
        # Route to appropriate workflow implementation
        if workflow_type == "dialog_only":
            result = await self._run_dialog_only_workflow(context)
        elif workflow_type == "dialog_only_diff":
            result = await self._run_dialog_only_diff_workflow(context)
        elif workflow_type == "differential_diagnosis":
            result = await self._run_differential_diagnosis_workflow(context)
        elif workflow_type == "super_agent_specialist_kb":
            result = await self._run_super_agent_specialist_kb_workflow(context)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Record execution time
        result.execution_time_seconds = round(time.time() - start_time, 2)
        
        # Print token usage summary
        self._print_token_summary()
        
        return result
    
    async def _run_dialog_only_workflow(self, context: WorkflowContext) -> CaseResult:
        """Dialog-only workflow: patient, doctor_q, diagnosis_dialog, judge.

        Patient and doctor interact each round. After every (non-IDK) patient
        answer, diagnosis_dialog_agent predicts from the raw dialog. Exits when
        the top prediction's confidence is >= high_confidence_threshold (config
        key, default from Settings), or when max_rounds is reached.
        """
        patient_agent = self.get_agent("patient_agent")
        doctor_q_agent = self.get_agent("doctor_q_agent")
        diagnosis_agent = self.get_agent("diagnosis_dialog_agent")
        judge_agent = self.get_agent("judge_agent")
        
        max_rounds = self.config.get("max_rounds", self.settings.max_dialog_rounds)
        high_conf_threshold = self.config.get("high_confidence_threshold", self.settings.high_confidence_threshold)
        
        final_diagnosis = None
        final_confidence = 0
        error_log: list[str] = []
        completion_status = "complete"
        
        for round_num in range(max_rounds):
            print(f"--- Round {round_num + 1}/{max_rounds} ---")
            # Doctor asks a question
            q_result = await doctor_q_agent.execute(
                context,
                current_round=round_num + 1,
                max_rounds=max_rounds
            )
            # question = input("Please enter your question: ")
            # q_result = type("DoctorQAgentOutput", (), {"success": True, "question": question})()  # Mocking doctor agent output
            if not q_result.success:
                error_log.append(f"Round {round_num+1}: doctor_q_agent failed - {q_result.error}")
                completion_status = "incomplete"
                break
            
            question = q_result.question
            q_reasoning = getattr(q_result, 'question_reasoning', None)
            context.add_doctor_question(question, question_reasoning=q_reasoning)
            context.set("current_question", question)
            context.add_trace("doctor_q_agent", "ask_question", {"question": question, "reasoning": q_reasoning})
            print(f"[Doctor] {question}")
            
            # Patient answers
            p_result = await patient_agent.execute(context)
            if not p_result.success:
                error_log.append(f"Round {round_num+1}: patient_agent failed - {p_result.error}")
                completion_status = "incomplete"
                break
            
            answer = p_result.answer
            print(f"[Patient] {answer}")
            
            # Check for "I don't know" - skip diagnosis and don't count this round
            if self._is_idk_answer(answer):
                # Add to dialog history (so same Q isn't asked again) but mark as IDK
                context.add_patient_idk_answer(answer)
                context.add_trace("patient_agent", "answer_idk", {"answer": answer, "skipped": True})
                continue
            
            context.add_patient_answer(answer)
            context.add_trace("patient_agent", "answer", {"answer": answer})
            
            # Skip diagnosis in early rounds if configured (for cost savings with expensive models)
            skip_diagnosis_until = self.config.get("skip_diagnosis_until", 0)
            if skip_diagnosis_until and round_num + 1 < skip_diagnosis_until:
                print(f"  [SKIP] Diagnosis skipped (round {round_num+1} < {skip_diagnosis_until})")
                continue
            
            # Make diagnosis
            anchor_preds = None
            diag_result = await diagnosis_agent.execute(context, previous_predictions=anchor_preds)
            if not diag_result.success or not diag_result.predictions:
                error_log.append(f"Round {round_num+1}: diagnosis_dialog_agent failed - {diag_result.error if not diag_result.success else 'no predictions'}")
                completion_status = "incomplete"
                break
            
            final_diagnosis = diag_result.disease
            final_confidence = diag_result.confidence
            
            all_preds = [{"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning} for p in diag_result.predictions]
            
            # Store predictions for next round's anchor
            
            anchored_label = " (anchored)" if anchor_preds else ""
            print(f"[Diagnosis{anchored_label}] {final_diagnosis} ({final_confidence}%)")
            context.update_last_turn_diagnosis(final_diagnosis, final_confidence)
            context.add_trace("diagnosis_dialog_agent", "diagnose", {
                "disease": final_diagnosis,
                "confidence": final_confidence,
                "reasoning": diag_result.reasoning,
                "final_predictions": all_preds,
                "anchored": anchor_preds is not None,
            })
            
            # Check exit condition
            if final_confidence >= high_conf_threshold:
                break
        
        # Judge the result
        if final_diagnosis:
            context.set("predicted_disease", final_diagnosis)
            judge_result = await judge_agent.execute(context)
            is_correct = judge_result.same if judge_result.success else False
            judge_explanation = judge_result.explanation if judge_result.success else None
            if not judge_result.success:
                error_log.append(f"judge_agent failed - {judge_result.error}")
        else:
            is_correct = False
            judge_explanation = None
            final_diagnosis = "No diagnosis"
        
        return CaseResult(
            case_id=context.get("case_id"),
            ground_truth=context.get("ground_truth"),
            predicted_disease=final_diagnosis,
            confidence=final_confidence,
            result="correct" if is_correct else "incorrect",
            judge_explanation=judge_explanation,
            rounds=context.get_round_count(),
            completion_status=completion_status,
            error_log=error_log if error_log else None,
            dialog_history=context.get_dialog_structured()
        )
    
    async def _run_dialog_only_diff_workflow(self, context: WorkflowContext) -> CaseResult:
        """Dialog-only workflow with differential questioning (no summarizer).

        Like the dialog-only workflow, but starting at round `diff_start_round`
        (config key, default 2) — once we have predictions from a prior round —
        switches to doctor_q_diff_agent, which is given the previous round's
        top predictions so it can ask targeted, discriminating questions.
        Optionally supports specialist ensemble (config `use_specialist_ensemble`,
        starting at `specialist_start_round`, default 5) and knowledge-graph
        candidate lookup (config `use_knowledge_graph`, starting at
        `kg_start_round`). No summarizer is used — diagnosis is made from the
        raw dialog. Exits when confidence >= high_confidence_threshold or
        max_rounds is reached.
        """
        patient_agent = self.get_agent("patient_agent")
        doctor_q_agent = self.get_agent("doctor_q_agent")
        doctor_q_diff_agent = self.get_agent("doctor_q_diff_agent")
        diagnosis_agent = self.get_agent("diagnosis_dialog_agent")
        judge_agent = self.get_agent("judge_agent")
        
        
        current_specialty = None
        
        
        use_knowledge_graph = self.config.get("use_knowledge_graph", False)
        kg_str_summary = None  # Structured summary for KG symptom extraction only
        if use_knowledge_graph:
            from src.knowledge_graphs import get_kb_disease_candidates, UMLSClient
            kg_summarizer = self.get_agent("summarizer_str_agent")
            diagnosis_kb_agent = self.get_agent("diagnosis_kb_agent")
            diagnosis_merge_agent = self.get_agent("diagnosis_merge_agent")
            kg_start_round = self.config.get("kg_start_round", 10)
            try:
                umls_client = UMLSClient()
            except ValueError as e:
                error_log_init = [f"UMLS client initialization failed: {e}"]
                umls_client = None
            else:
                error_log_init = []
        
        use_evidence_gap = self.config.get("use_evidence_gap", False)
        evidence_gap_report = None
        general_gap_report = None
        if use_evidence_gap:
            evidence_gap_agent = self.get_agent("evidence_gap_agent")
            doctor_q_diff_evidence_agent = self.get_agent("doctor_q_diff_evidence_agent")
            eg_start_round = self.config.get("eg_start_round", 10)
        
        use_early_evidence_gap = self.config.get("use_early_evidence_gap", False)
        if use_early_evidence_gap:
            if not use_evidence_gap:
                evidence_gap_agent = self.get_agent("evidence_gap_agent")
            doctor_q_evidence_agent = self.get_agent("doctor_q_evidence_agent")
            early_eg_start_round = self.config.get("early_eg_start_round", 2)
        
        max_rounds = self.config.get("max_rounds", self.settings.max_dialog_rounds)
        high_conf_threshold = self.config.get("high_confidence_threshold", self.settings.high_confidence_threshold)
        
        final_diagnosis = None
        final_confidence = 0
        current_predictions = None
        error_log: list[str] = []
        if use_knowledge_graph:
            error_log.extend(error_log_init)
        completion_status = "complete"
        diff_start_round = self.config.get("diff_start_round", 2)

        # Counter for the meaningful-stopping heuristic (used below to require
        # several consecutive high-confidence rounds before exiting).
        meaningful_count = 0

        # Specialist Ensemble mode (A3b)
        use_specialist_ensemble = self.config.get("use_specialist_ensemble", False)
        if use_specialist_ensemble:
            from src.agents.specialist_ensemble_agent import SPECIALTIES
            specialist_ensemble_agent = self.get_agent("specialist_ensemble_agent")
            specialist_start_round = self.config.get("specialist_start_round", 5)
        
        for round_num in range(max_rounds):
            print(f"--- Round {round_num + 1}/{max_rounds} ---")
            
            # Determine which agent to use and generate question
            use_diff = round_num + 1 >= diff_start_round and current_predictions is not None
            feedback = None
            rejected_question = None
            question = None
            
            retry_limit = 0
            for retry_num in range(retry_limit + 1):
                # Variant mode: use variant diff agent
                if not use_diff:
                    # Early evidence gap: use doctor_q_evidence_agent if gap report available
                    if use_early_evidence_gap and general_gap_report and round_num + 1 >= early_eg_start_round:
                        q_result = await doctor_q_evidence_agent.execute(
                            context,
                            evidence_gap_report=general_gap_report,
                            current_round=round_num + 1,
                            max_rounds=max_rounds
                        )
                        if not q_result.success:
                            error_log.append(f"Round {round_num+1}: doctor_q_evidence_agent failed - {q_result.error}")
                            # Fall back to regular doctor_q_agent
                            q_result = await doctor_q_agent.execute(
                                context,
                                current_round=round_num + 1,
                                max_rounds=max_rounds,
                                feedback=feedback,
                                rejected_question=rejected_question
                            )
                            if not q_result.success:
                                error_log.append(f"Round {round_num+1}: doctor_q_agent fallback failed - {q_result.error}")
                                break
                        question = q_result.question
                    else:
                        q_result = await doctor_q_agent.execute(
                            context,
                            current_round=round_num + 1,
                            max_rounds=max_rounds,
                            feedback=feedback,
                            rejected_question=rejected_question
                        )
                        if not q_result.success:
                            error_log.append(f"Round {round_num+1}: doctor_q_agent failed - {q_result.error}")
                            break
                        question = q_result.question
                else:
                    predictions_for_diff = current_predictions  # already dicts
                    # Use evidence gap agent if enabled and report available
                    if use_evidence_gap and evidence_gap_report and round_num + 1 >= eg_start_round:
                        q_diff_result = await doctor_q_diff_evidence_agent.execute(
                            context,
                            predictions=predictions_for_diff,
                            evidence_gap_report=evidence_gap_report,
                            str_summary=context.get_dialog_readable(),
                            current_round=round_num + 1,
                            max_rounds=max_rounds
                        )
                    else:
                        q_diff_result = await doctor_q_diff_agent.execute(
                            context,
                            predictions=predictions_for_diff,
                            str_summary=context.get_dialog_readable(),
                            current_round=round_num + 1,
                            max_rounds=max_rounds,
                            feedback=feedback,
                            rejected_question=rejected_question
                        )
                    if not q_diff_result.success:
                        error_log.append(f"Round {round_num+1}: doctor_q_diff_agent failed - {q_diff_result.error}")
                        # Fall back to regular doctor_q_agent
                        q_result = await doctor_q_agent.execute(
                            context,
                            current_round=round_num + 1,
                            max_rounds=max_rounds,
                            feedback=feedback,
                            rejected_question=rejected_question
                        )
                        if not q_result.success:
                            error_log.append(f"Round {round_num+1}: doctor_q_agent fallback failed - {q_result.error}")
                            break
                        question = q_result.question
                        context.add_trace("doctor_q_agent", "ask_question_fallback", {"question": question, "round": round_num + 1})
                    else:
                        question = q_diff_result.question
                
                if not question:
                    break
                
                # Check for repetition if enabled (skip on last retry)
                break
            
            if not question:
                completion_status = "incomplete"
                break
            
            # Log trace for the accepted question
            if use_diff:
                context.add_trace("doctor_q_diff_agent", "ask_differential_question", {
                    "question": question, "round": round_num + 1
                })
            else:
                context.add_trace("doctor_q_agent", "ask_question", {"question": question, "round": round_num + 1})
            
            context.add_doctor_question(question)
            context.set("current_question", question)
            print(f"[Doctor] {question}")
            
            # Patient answers
            p_result = await patient_agent.execute(context)
            if not p_result.success:
                error_log.append(f"Round {round_num+1}: patient_agent failed - {p_result.error}")
                completion_status = "incomplete"
                break
            
            answer = p_result.answer
            print(f"[Patient] {answer}")
            
            # Check for IDK
            if self._is_idk_answer(answer):
                context.add_patient_idk_answer(answer)
                context.add_trace("patient_agent", "answer_idk", {"answer": answer, "skipped": True})
                # Update KG structured summary on IDK too
                if use_knowledge_graph:
                    dialog_lastturn = f"D: {question}\nP: {answer}"
                    kg_sum_result = await kg_summarizer.execute(context, dialog_lastturn=dialog_lastturn, str_summary=kg_str_summary)
                    if kg_sum_result.success:
                        kg_str_summary = kg_sum_result.str_summary.model_dump()
                continue
            
            context.add_patient_answer(answer)
            context.add_trace("patient_agent", "answer", {"answer": answer})
            
            # Update KG structured summary (for symptom extraction only)
            if use_knowledge_graph:
                dialog_lastturn = f"D: {question}\nP: {answer}"
                kg_sum_result = await kg_summarizer.execute(context, dialog_lastturn=dialog_lastturn, str_summary=kg_str_summary)
                if kg_sum_result.success:
                    kg_str_summary = kg_sum_result.str_summary.model_dump()
                else:
                    error_log.append(f"Round {round_num+1}: kg_summarizer failed - {kg_sum_result.error}")
            
            # Classify specialty if enabled and past start round
            
            # Skip diagnosis in early rounds if configured (for cost savings with expensive models)
            skip_diagnosis_until = self.config.get("skip_diagnosis_until", 0)
            if skip_diagnosis_until and round_num + 1 < skip_diagnosis_until:
                print(f"  [SKIP] Diagnosis skipped (round {round_num+1} < {skip_diagnosis_until})")
                continue
            
            # Make diagnosis from dialog
            # Specialist Ensemble mode: run all 10 specialists + general, then merge
            if use_specialist_ensemble and round_num + 1 >= specialist_start_round:
                specialist_preds: dict[str, list[dict]] = {}
                dialog_readable = context.get_dialog_readable()
                
                # Run general (no specialty) first
                gen_result = await diagnosis_agent.execute(context, specialty=None)
                if gen_result.success and gen_result.predictions:
                    specialist_preds["General Physician"] = [
                        {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                        for p in gen_result.predictions
                    ]
                    print(f"  [SPECIALIST] General: {gen_result.disease} ({gen_result.confidence}%)")
                else:
                    error_log.append(f"Round {round_num+1}: diagnosis (general) failed")
                
                # Run each specialist sequentially
                for spec in SPECIALTIES:
                    spec_result = await diagnosis_agent.execute(context, specialty=spec)
                    if spec_result.success and spec_result.predictions:
                        specialist_preds[spec] = [
                            {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                            for p in spec_result.predictions
                        ]
                        print(f"  [SPECIALIST] {spec}: {spec_result.disease} ({spec_result.confidence}%)")
                    else:
                        error_log.append(f"Round {round_num+1}: diagnosis ({spec}) failed")
                
                # Merge via ensemble agent
                if specialist_preds:
                    # Log compact per-turn specialist predictions (disease + confidence only)
                    _compact_specialist_preds = {
                        spec: [{"disease": p["disease"], "confidence": p["confidence"]} for p in preds]
                        for spec, preds in specialist_preds.items()
                    }
                    context.update_last_turn_specialist_predictions(
                        json.dumps(_compact_specialist_preds, separators=(",", ":"))
                    )
                    ensemble_result = await specialist_ensemble_agent.execute(
                        context,
                        specialist_predictions=specialist_preds,
                        dialog=dialog_readable
                    )
                    if ensemble_result.success and ensemble_result.predictions:
                        diag_result = ensemble_result
                        context.add_trace("specialist_ensemble_agent", "merge", {
                            "specialist_predictions": specialist_preds,
                        })
                    else:
                        error_log.append(f"Round {round_num+1}: specialist_ensemble_agent failed - {ensemble_result.error}")
                        # Fall back to general predictions
                        diag_result = gen_result
                else:
                    error_log.append(f"Round {round_num+1}: all specialist diagnoses failed")
                    continue
            else:
                diag_result = await diagnosis_agent.execute(context, specialty=current_specialty)
            
            if not diag_result.success or not diag_result.predictions:
                error_log.append(f"Round {round_num+1}: diagnosis_dialog_agent failed - {diag_result.error if not diag_result.success else 'no predictions'}")
                completion_status = "incomplete"
                break
            
            # Store predictions for differential questioning
            current_predictions = diag_result.predictions
            
            final_diagnosis = diag_result.disease
            final_confidence = diag_result.confidence
            print(f"[Diagnosis] {final_diagnosis} ({final_confidence}%)")
            
            all_predictions_formatted = [
                {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                for p in diag_result.predictions
            ]
            
            context.update_last_turn_differential_diagnosis(
                disease=final_diagnosis,
                confidence=final_confidence,
                final_predictions=all_predictions_formatted
            )
            context.update_last_turn_prediction_status("success")
            
            context.add_trace("diagnosis_dialog_agent", "diagnose", {
                "disease": final_diagnosis,
                "confidence": final_confidence,
                "final_predictions": all_predictions_formatted
            })
            
            # Critique + Reflect flow if enabled
            
            # Run KG pipeline if enabled and past start round
            if use_knowledge_graph and umls_client is not None and kg_str_summary is not None and round_num + 1 >= kg_start_round:
                try:
                    kb_candidates, kb_candidates_formatted = get_kb_disease_candidates(
                        kg_str_summary, umls_client, max_candidates=10000
                    )
                except Exception as e:
                    kb_candidates = []
                    error_log.append(f"Round {round_num+1}: KG lookup failed - {str(e)}")
                
                if kb_candidates:
                    use_dialog_for_kb = self.config.get("use_dialog_for_kb", False)
                    if use_dialog_for_kb:
                        kb_result = await diagnosis_kb_agent.execute(
                            context, dialog=context.get_dialog_readable(), kb_candidates=kb_candidates_formatted
                        )
                    else:
                        kb_result = await diagnosis_kb_agent.execute(
                            context, str_summary=kg_str_summary, kb_candidates=kb_candidates_formatted
                        )
                    if kb_result.success and kb_result.predictions:
                        # Merge KB + LLM predictions using dialog
                        kb_preds = [{"disease": p.disease, "reasoning": p.reasoning} for p in kb_result.predictions]
                        llm_preds = [{"disease": p["disease"], "reasoning": p.get("reasoning", "")} for p in all_predictions_formatted]
                        merge_result = await diagnosis_merge_agent.execute(
                            context,
                            dialog=context.get_dialog_readable(),
                            kb_predictions=kb_preds,
                            llm_predictions=llm_preds
                        )
                        if merge_result.success and merge_result.predictions:
                            # Override predictions with merged results
                            all_predictions_formatted = [
                                {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                                for p in merge_result.predictions
                            ]
                            final_diagnosis = all_predictions_formatted[0]["disease"]
                            final_confidence = all_predictions_formatted[0]["confidence"]
                            print(f"  [KG_MERGE] {final_diagnosis} ({final_confidence}%)")
                        else:
                            error_log.append(f"Round {round_num+1}: diagnosis_merge_agent failed")
                    else:
                        error_log.append(f"Round {round_num+1}: diagnosis_kb_agent failed")
            
            # Run evidence gap analysis if enabled and past start round
            if use_evidence_gap and round_num + 1 >= eg_start_round:
                gap_result = await evidence_gap_agent.execute(
                    context,
                    predictions=all_predictions_formatted,
                    dialog=context.get_dialog_readable()
                )
                if gap_result.success:
                    evidence_gap_report = gap_result.format_for_doctor()
                    print(f"  [EVIDENCE_GAP] Completed")
                    _general_str = "\n".join(gap_result.report.general_gaps) if gap_result.report.general_gaps else None
                    _diff_str = "\n".join(
                        f"{d.disease}: {d.missing_evidence}" for d in gap_result.report.diagnosis_specific_gaps
                    ) if gap_result.report.diagnosis_specific_gaps else None
                    context.update_last_turn_evidence_gaps(general=_general_str, differential=_diff_str)
                else:
                    error_log.append(f"Round {round_num+1}: evidence_gap_agent failed - {gap_result.error}")
            
            # Run general evidence gap for early rounds (before diff starts)
            # Generate one round before early_eg_start_round so report is ready when doctor_q_evidence_agent needs it
            if use_early_evidence_gap and round_num + 1 >= early_eg_start_round - 1 and round_num + 1 < diff_start_round:
                gen_gap_result = await evidence_gap_agent.execute_general(
                    context,
                    dialog=context.get_dialog_readable()
                )
                if gen_gap_result.success:
                    general_gap_report = gen_gap_result.format_for_doctor()
                    print(f"  [GENERAL_GAP] {len(gen_gap_result.report.general_gaps)} gaps identified")
                    _general_str = "\n".join(gen_gap_result.report.general_gaps) if gen_gap_result.report.general_gaps else None
                    context.update_last_turn_evidence_gaps(general=_general_str)
                else:
                    error_log.append(f"Round {round_num+1}: evidence_gap_agent (general) failed - {gen_gap_result.error}")
            
            # Store predictions for differential questioning (may be CF-merged)
            current_predictions = all_predictions_formatted
            
            # Variant mode trigger: if confidence >= threshold and round >= variant_start_round
            
            # Meaningful stopping: track non-IDK rounds and check for info exhaustion
            
            # Check exit condition
            if final_confidence >= high_conf_threshold:
                break
        
        # Log meaningful rounds if applicable
        
        # Judge the result
        if final_diagnosis:
            context.set("predicted_disease", final_diagnosis)
            judge_result = await judge_agent.execute(context)
            is_correct = judge_result.same if judge_result.success else False
            judge_explanation = judge_result.explanation if judge_result.success else None
            if not judge_result.success:
                error_log.append(f"judge_agent failed - {judge_result.error}")
        else:
            is_correct = False
            judge_explanation = None
            final_diagnosis = "No diagnosis"
        
        return CaseResult(
            case_id=context.get("case_id"),
            ground_truth=context.get("ground_truth"),
            predicted_disease=final_diagnosis,
            confidence=final_confidence,
            result="correct" if is_correct else "incorrect",
            judge_explanation=judge_explanation,
            rounds=context.get_round_count(),
            completion_status=completion_status,
            error_log=error_log if error_log else None,
            dialog_history=context.get_dialog_structured()
        )


    
    
    
    async def _run_differential_diagnosis_workflow(self, context: WorkflowContext) -> CaseResult:
        """Differential diagnosis workflow: patient, doctor_q[_diff], summarizer, diagnosis_summary, judge.

        Uses doctor_q_diff_agent to ask targeted questions that discriminate between
        the top 3 differential diagnoses. Initial rounds use doctor_q_agent; once
        predictions are available and round_num >= diff_start_round (config key,
        default 2 for this workflow), switches to doctor_q_diff_agent which is
        given all 3 predictions. Diagnosis is made by diagnosis_summary_agent
        from the case summary; the summary can be paragraph form
        (`summary_type='default'`) or structured (`summary_type='structured'`,
        the default). Exits when confidence >= high_confidence_threshold or
        max_rounds is reached.
        """
        patient_agent = self.get_agent("patient_agent")
        doctor_q_agent = self.get_agent("doctor_q_agent")  # For initial rounds
        doctor_q_diff_agent = self.get_agent("doctor_q_diff_agent")  # For differential questions
        diagnosis_agent = self.get_agent("diagnosis_summary_agent")
        judge_agent = self.get_agent("judge_agent")
        
        summary_type = self.config.get("summary_type", "structured")
        use_structured = summary_type == "structured"
        if use_structured:
            print("[Workflow] Structured summarizer agent")
            summarizer_agent = self.get_agent("summarizer_str_agent")
        else:
            print("[Workflow] Paragraph summarizer agent")
            summarizer_agent = self.get_agent("summarizer_agent")
        
        max_rounds = self.config.get("max_rounds", self.settings.max_dialog_rounds)
        high_conf_threshold = self.config.get("high_confidence_threshold", self.settings.high_confidence_threshold)
        diff_start_round = self.config.get("diff_start_round")
        pass_summary_to_doctor = self.config.get("pass_summary_to_doctor", False)
        # if pass_summary_to_doctor:
        #     print("[Workflow] Passing summary to doctor")
        # else:
        #     print("[Workflow] Not passing summary to doctor, doctor only sees dialog")
        
        final_diagnosis = None
        final_confidence = 0
        str_summary = None  # Structured summary (dict), only used for structured mode
        final_summary = ""  # String version for output
        current_predictions = None  # Store top 3 predictions for differential questioning
        error_log: list[str] = []
        completion_status = "complete"
        
        for round_num in range(max_rounds):
            print(f"--- Round {round_num + 1}/{max_rounds} ---")
            # Determine which agent to use for questioning
            if round_num + 1 < diff_start_round or current_predictions is None:
                # Initial rounds: use regular doctor_q_agent
                q_result = await doctor_q_agent.execute(
                    context,
                    current_round=round_num + 1,
                    max_rounds=max_rounds
                )
                if not q_result.success:
                    error_log.append(f"Round {round_num+1}: doctor_q_agent failed - {q_result.error}")
                    completion_status = "incomplete"
                    break
                question = q_result.question
                context.add_trace("doctor_q_agent", "ask_question", {"question": question, "round": round_num + 1})
            else:
                # Subsequent rounds: use doctor_q_diff_agent with top 3 predictions
                predictions_for_diff = [
                    {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                    for p in current_predictions
                ]
                # Pass dialog or summary to diff agent depending on config
                diff_context = final_summary if (pass_summary_to_doctor and final_summary) else context.get_dialog_readable()
                q_diff_result = await doctor_q_diff_agent.execute(
                    context,
                    predictions=predictions_for_diff,
                    str_summary=diff_context,
                    current_round=round_num + 1,
                    max_rounds=max_rounds
                )
                if not q_diff_result.success:
                    error_log.append(f"Round {round_num+1}: doctor_q_diff_agent failed - {q_diff_result.error}")
                    # Fall back to regular doctor_q_agent
                    q_result = await doctor_q_agent.execute(
                        context,
                        current_round=round_num + 1,
                        max_rounds=max_rounds
                    )
                    if not q_result.success:
                        error_log.append(f"Round {round_num+1}: doctor_q_agent fallback failed - {q_result.error}")
                        completion_status = "incomplete"
                        break
                    question = q_result.question
                    context.add_trace("doctor_q_agent", "ask_question_fallback", {"question": question, "round": round_num + 1})
                else:
                    question = q_diff_result.question
                    context.add_trace("doctor_q_diff_agent", "ask_differential_question", {
                        "question": question,
                        "round": round_num + 1,
                        "differential_predictions": predictions_for_diff
                    })
            
            context.add_doctor_question(question)
            context.set("current_question", question)
            print(f"[Doctor] {question}")
            
            # Patient answers
            p_result = await patient_agent.execute(context)
            if not p_result.success:
                error_log.append(f"Round {round_num+1}: patient_agent failed - {p_result.error}")
                completion_status = "incomplete"
                break
            
            answer = p_result.answer
            print(f"[Patient] {answer}")
            
            # Check for "I don't know" - update summary but skip diagnosis
            if self._is_idk_answer(answer):
                context.add_patient_idk_answer(answer)
                context.add_trace("patient_agent", "answer_idk", {"answer": answer, "skipped": True})
                
                # For structured summary, still update to capture uncertainty
                if use_structured:
                    dialog_lastturn = f"D: {question}\nP: {answer}"
                    summary_result = await summarizer_agent.execute(
                        context,
                        dialog_lastturn=dialog_lastturn,
                        str_summary=str_summary
                    )
                    if summary_result.success:
                        str_summary = summary_result.str_summary.model_dump()
                        import json
                        final_summary = json.dumps(str_summary, indent=2)
                        context.set("summary", final_summary)
                    else:
                        error_log.append(f"Round {round_num+1}: summarizer failed - {summary_result.error}")
                
                continue
            
            context.add_patient_answer(answer)
            context.add_trace("patient_agent", "answer", {"answer": answer})
            
            # Generate/update summary
            if use_structured:
                dialog_lastturn = f"D: {question}\nP: {answer}"
                summary_result = await summarizer_agent.execute(
                    context,
                    dialog_lastturn=dialog_lastturn,
                    str_summary=str_summary
                )
                if summary_result.success:
                    str_summary = summary_result.str_summary.model_dump()
                    import json
                    final_summary = json.dumps(str_summary, indent=2)
                    context.set("summary", final_summary)
                else:
                    error_log.append(f"Round {round_num+1}: summarizer_str_agent failed - {summary_result.error}")
            else:
                summary_result = await summarizer_agent.execute(context)
                if summary_result.success:
                    final_summary = summary_result.summary
                    context.set("summary", final_summary)
                else:
                    error_log.append(f"Round {round_num+1}: summarizer_agent failed - {summary_result.error}")
            
            # Make diagnosis from summary
            diag_result = await diagnosis_agent.execute(context)
            if not diag_result.success or not diag_result.predictions:
                error_msg = diag_result.error if hasattr(diag_result, 'error') and diag_result.error else "Empty predictions"
                context.update_last_turn_prediction_status(f"failed: {error_msg}")
                context.add_trace("diagnosis_summary_agent", "diagnose_failed", {
                    "success": diag_result.success,
                    "error": diag_result.error if hasattr(diag_result, 'error') else None
                })
                error_log.append(f"Round {round_num+1}: diagnosis_summary_agent failed - {diag_result.error if not diag_result.success else 'no predictions'}")
                continue
            
            # Store all predictions for next round's differential questioning
            current_predictions = diag_result.predictions
            
            final_diagnosis = diag_result.disease
            final_confidence = diag_result.confidence
            
            # Format all predictions for storage (include reasoning)
            all_predictions_formatted = [
                {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                for p in diag_result.predictions
            ]
            
            # Update last turn with all 3 predictions and success status
            context.update_last_turn_differential_diagnosis(
                disease=final_diagnosis,
                confidence=final_confidence,
                final_predictions=all_predictions_formatted
            )
            context.update_last_turn_prediction_status("success")
            
            context.add_trace("diagnosis_summary_agent", "diagnose", {
                "disease": final_diagnosis,
                "confidence": final_confidence,
                "final_predictions": all_predictions_formatted
            })
            
            # Check exit condition
            if final_confidence >= high_conf_threshold:
                break
        
        # Judge the result
        if final_diagnosis:
            context.set("predicted_disease", final_diagnosis)
            judge_result = await judge_agent.execute(context)
            is_correct = judge_result.same if judge_result.success else False
            judge_explanation = judge_result.explanation if judge_result.success else None
            if not judge_result.success:
                error_log.append(f"judge_agent failed - {judge_result.error}")
        else:
            is_correct = False
            judge_explanation = None
            final_diagnosis = "No diagnosis"
        
        return CaseResult(
            case_id=context.get("case_id"),
            ground_truth=context.get("ground_truth"),
            predicted_disease=final_diagnosis,
            confidence=final_confidence,
            result="correct" if is_correct else "incorrect",
            judge_explanation=judge_explanation,
            rounds=context.get_round_count(),
            completion_status=completion_status,
            error_log=error_log if error_log else None,
            dialog_history=context.get_dialog_structured(),
            summary=final_summary
        )
    
    
    

    





    async def _run_super_agent_specialist_kb_workflow(self, context: WorkflowContext) -> CaseResult:
        """Full MeDxAgent pipeline: specialist ensemble + KB + evidence gaps.

        Reference variant: ``v_p12_f2_turn10_A1a3b56b`` — combines
        paragraph summary, specialist ensemble, knowledge-graph
        candidate lookup, and evidence-gap-guided questioning.

        All round thresholds below are configurable; the values shown are the
        defaults used by the flagship variant.

        Early rounds (round < agent_pipeline_start_round, default 10):
          1. doctor_q_agent asks a question — or doctor_q_evidence_agent once
             round >= early_eg_start_round (default 3) and a general gap report
             is available from the previous round.
          2. patient_agent answers.
          3. summarizer_str_agent (structured) updates the symptom summary;
             summarizer_agent (paragraph) also updates if use_paragraph_summary.
          4. diagnosis_dialog_agent predicts from the raw dialog (used only for
             early-exit checks, not stored as the final answer).
          5. evidence_gap_agent.execute_general() refreshes the general gap
             report for the next round (when round >= early_eg_start_round - 1).

        Full-pipeline rounds (round >= agent_pipeline_start_round):
          1. doctor_q_diff_evidence_agent asks a discriminating, evidence-gap-
             guided question (or doctor_q_diff_agent when use_evidence_gap is
             False), given the previous round's merged predictions.
          2. patient_agent answers.
          3. Both summarizers update as above.
          4. specialist_diagnosis_agent is called once per specialty plus a
             general call (10 calls total — 9 specialties + 1 general); then
             specialist_ensemble_agent merges the per-specialist predictions
             into 3 ensemble predictions.
          5. get_kb_disease_candidates() looks up UMLS candidates from the
             structured symptom summary; diagnosis_kb_agent picks the top 3.
          6. diagnosis_merge_agent merges the 3 ensemble + 3 KB predictions
             into the final 3 predictions for the round.
          7. evidence_gap_agent.execute() generates differential + general
             gap reports to drive the next round's questions.
          8. Exits when the top prediction's confidence is
             >= high_confidence_threshold (default 95), or when max_rounds is
             reached.
        """
        from src.knowledge_graphs import get_kb_disease_candidates, UMLSClient
        from src.agents.specialist_ensemble_agent import SPECIALTIES
        import json
        
        patient_agent = self.get_agent("patient_agent")
        doctor_q_agent = self.get_agent("doctor_q_agent")
        use_evidence_gap = self.config.get("use_evidence_gap", True)
        doctor_q_evidence_agent = self.get_agent("doctor_q_evidence_agent") if use_evidence_gap else None
        doctor_q_diff_evidence_agent = self.get_agent("doctor_q_diff_evidence_agent") if use_evidence_gap else None
        doctor_q_diff_agent = self.get_agent("doctor_q_diff_agent") if not use_evidence_gap else None
        summarizer_str_agent = self.get_agent("summarizer_str_agent")  # structured (always)
        use_paragraph_summary = self.config.get("use_paragraph_summary", False)
        summarizer_agent = self.get_agent("summarizer_agent") if use_paragraph_summary else None  # paragraph (optional)
        diagnosis_dialog_agent = self.get_agent("diagnosis_dialog_agent")  # for early rounds (and specialists when no summary)
        diagnosis_summary_agent = self.get_agent("diagnosis_summary_agent") if use_paragraph_summary else None  # for specialist ensemble with summary
        # Agent used for specialist calls: summary agent when paragraph summary enabled, dialog agent otherwise
        specialist_diagnosis_agent = diagnosis_summary_agent if use_paragraph_summary else diagnosis_dialog_agent
        use_specialist_ensemble = self.config.get("use_specialist_ensemble", True)
        specialist_ensemble_agent = self.get_agent("specialist_ensemble_agent") if use_specialist_ensemble else None
        use_knowledge_graph = self.config.get("use_knowledge_graph", True)
        diagnosis_kb_agent = self.get_agent("diagnosis_kb_agent") if use_knowledge_graph else None
        diagnosis_merge_agent = self.get_agent("diagnosis_merge_agent") if use_knowledge_graph else None
        evidence_gap_agent = self.get_agent("evidence_gap_agent") if use_evidence_gap else None
        judge_agent = self.get_agent("judge_agent")
        
        early_eg_start_round = self.config.get("early_eg_start_round", 3)
        
        max_rounds = self.config.get("max_rounds", self.settings.max_dialog_rounds)
        high_conf_threshold = self.config.get("high_confidence_threshold", self.settings.high_confidence_threshold)
        diff_start_round = self.config.get("diff_start_round", 11)
        agent_pipeline_start_round = diff_start_round - 1  # default: 10

        # =====================================================================
        # QUALITATIVE PRINT HELPERS (for paper-ready agent output traces)
        # Enable in variant YAML with `qualitative_print: true`
        # =====================================================================
        qualitative_print = self.config.get("qualitative_print", False)

        def _qp(text: str = "") -> None:
            if qualitative_print:
                print(text)

        def _qp_section(title: str) -> None:
            if not qualitative_print:
                return
            print()
            print("─" * 80)
            print(f"  {title}")
            print("─" * 80)

        def _qp_round_banner(n: int, total: int) -> None:
            if not qualitative_print:
                return
            print()
            print("=" * 80)
            print(f"  ROUND {n} / {total}")
            print("=" * 80)

        def _qp_predictions(label: str, preds: list[dict]) -> None:
            if not qualitative_print or not preds:
                return
            _qp_section(label)
            for i, p in enumerate(preds, 1):
                disease = p.get("disease", "")
                conf = p.get("confidence", None)
                reasoning = p.get("reasoning", "")
                conf_str = f"  ({conf}%)" if conf is not None and conf != "" else ""
                _qp(f"  {i}. {disease}{conf_str}")
                if reasoning:
                    _qp(f"     Reasoning: {reasoning}")

        def _qp_pred_objs(label: str, preds_objs) -> None:
            """Same as _qp_predictions but accepts pydantic Prediction objects."""
            if not qualitative_print or not preds_objs:
                return
            _qp_section(label)
            for i, p in enumerate(preds_objs, 1):
                disease = getattr(p, "disease", "")
                conf = getattr(p, "confidence", None)
                reasoning = getattr(p, "reasoning", "")
                conf_str = f"  ({conf}%)" if conf is not None and conf != "" else ""
                _qp(f"  {i}. {disease}{conf_str}")
                if reasoning:
                    _qp(f"     Reasoning: {reasoning}")
        
        final_diagnosis = None
        final_confidence = 0
        str_summary = None
        para_summary = None
        final_summary = ""
        current_predictions = None
        evidence_gap_report = None
        general_gap_report = None
        error_log: list[str] = []
        completion_status = "complete"
        
        final_llm_predictions = None
        final_kb_predictions = None
        
        # Initialize UMLS client (only if KB is enabled)
        umls_client = None
        if use_knowledge_graph:
            try:
                umls_client = UMLSClient()
            except ValueError as e:
                error_log.append(f"UMLS client initialization failed: {e}")
                print(f"  [UMLS_ERROR] Client initialization failed: {e}")
                umls_client = None
        
        for round_num in range(max_rounds):
            round_display = round_num + 1
            print(f"--- Round {round_display}/{max_rounds} ---")
            _qp_round_banner(round_display, max_rounds)
            
            # Reset per-turn token counters
            for agent in self._agents.values():
                agent.reset_turn_tokens()
            
            # ============================================================
            # STEP 1: DOCTOR ASKS QUESTION
            # ============================================================
            use_diff = round_display >= diff_start_round and current_predictions is not None
            
            if not use_diff:
                if round_display >= early_eg_start_round and general_gap_report:
                    q_result = await doctor_q_evidence_agent.execute(
                        context,
                        evidence_gap_report=general_gap_report,
                        current_round=round_display,
                        max_rounds=max_rounds
                    )
                    if not q_result.success:
                        error_log.append(f"Round {round_display}: doctor_q_evidence_agent failed - {q_result.error}")
                        completion_status = "incomplete"
                        break
                    question = q_result.question
                    context.add_trace("doctor_q_evidence_agent", "ask_evidence_guided_question", {
                        "question": question, "round": round_display
                    })
                    _qp_section("DOCTOR Q (evidence-gap guided)")
                    _qp(f"  Question: {question}")
                    _qr = getattr(q_result, "question_reasoning", None)
                    if _qr:
                        _qp(f"  Reasoning: {_qr}")
                else:
                    # print("Using general doctor, not evidence gap doctor")
                    q_result = await doctor_q_agent.execute(
                        context,
                        current_round=round_display,
                        max_rounds=max_rounds
                    )
                    if not q_result.success:
                        error_log.append(f"Round {round_display}: doctor_q_agent failed - {q_result.error}")
                        completion_status = "incomplete"
                        break
                    question = q_result.question
                    context.add_trace("doctor_q_agent", "ask_question", {"question": question, "round": round_display})
                    _qp_section("DOCTOR Q (general)")
                    _qp(f"  Question: {question}")
                    _qr = getattr(q_result, "question_reasoning", None)
                    if _qr:
                        _qp(f"  Reasoning: {_qr}")
            else:
                predictions_for_diff = current_predictions
                if use_evidence_gap:
                    q_diff_result = await doctor_q_diff_evidence_agent.execute(
                        context,
                        predictions=predictions_for_diff,
                        evidence_gap_report=evidence_gap_report,
                        str_summary=context.get_dialog_readable(),
                        current_round=round_display,
                        max_rounds=max_rounds
                    )
                    if not q_diff_result.success:
                        error_log.append(f"Round {round_display}: doctor_q_diff_evidence_agent failed - {q_diff_result.error}")
                        completion_status = "incomplete"
                        break
                    question = q_diff_result.question
                    context.add_trace("doctor_q_diff_evidence_agent", "ask_evidence_gap_question", {
                        "question": question, "round": round_display
                    })
                    _qp_section("DOCTOR Q (differential + evidence-gap)")
                    _qp(f"  Question: {question}")
                    _qr = getattr(q_diff_result, "question_reasoning", None)
                    if _qr:
                        _qp(f"  Reasoning: {_qr}")
                else:
                    q_diff_result = await doctor_q_diff_agent.execute(
                        context,
                        predictions=predictions_for_diff,
                        str_summary=context.get_dialog_readable(),
                        current_round=round_display,
                        max_rounds=max_rounds
                    )
                    if not q_diff_result.success:
                        error_log.append(f"Round {round_display}: doctor_q_diff_agent failed - {q_diff_result.error}")
                        completion_status = "incomplete"
                        break
                    question = q_diff_result.question
                    context.add_trace("doctor_q_diff_agent", "ask_diff_question", {
                        "question": question, "round": round_display
                    })
                    _qp_section("DOCTOR Q (differential)")
                    _qp(f"  Question: {question}")
            
            context.add_doctor_question(question)
            context.set("current_question", question)
            print(f"[Doctor] {question}")
            
            # ============================================================
            # STEP 2: PATIENT ANSWERS
            # ============================================================
            p_result = await patient_agent.execute(context)
            if not p_result.success:
                error_log.append(f"Round {round_display}: patient_agent failed - {p_result.error}")
                completion_status = "incomplete"
                break
            
            answer = p_result.answer
            print(f"[Patient] {answer}")
            _qp_section("PATIENT")
            _qp(f"  Answer: {answer}")
            
            if self._is_idk_answer(answer):
                context.add_patient_idk_answer(answer)
                context.add_trace("patient_agent", "answer_idk", {"answer": answer, "skipped": True})
                self._print_turn_tokens(round_display)
                continue
            
            context.add_patient_answer(answer)
            context.add_trace("patient_agent", "answer", {"answer": answer})
            
            # ============================================================
            # STEP 3: UPDATE BOTH SUMMARIES
            # ============================================================
            dialog_lastturn = f"D: {question}\nP: {answer}"
            
            # Structured summary (for KB symptom extraction)
            summary_result = await summarizer_str_agent.execute(
                context, dialog_lastturn=dialog_lastturn, str_summary=str_summary
            )
            if summary_result.success:
                str_summary = summary_result.str_summary.model_dump()
                if qualitative_print:
                    _qp_section("STRUCTURED SUMMARY (summarizer_str_agent)")
                    _qp(json.dumps(str_summary, indent=2))
            else:
                error_log.append(f"Round {round_display}: summarizer_str_agent failed - {summary_result.error}")
            
            # Paragraph summary (for diagnosis/merge/evidence gap agents when enabled)
            if summarizer_agent:
                para_result = await summarizer_agent.execute(context)
                if para_result.success:
                    para_summary = para_result.summary
                    context.set("summary", para_summary)
                    final_summary = para_summary
                    _qp_section("PARAGRAPH SUMMARY (summarizer_agent)")
                    _qp(f"  {para_summary}")
                else:
                    error_log.append(f"Round {round_display}: summarizer_agent failed - {para_result.error}")
            else:
                # No paragraph summarizer — use structured summary for context
                final_summary = json.dumps(str_summary, indent=2) if str_summary else ""
                context.set("summary", final_summary)
            
            # ============================================================
            # EARLY ROUNDS: DIAGNOSIS FROM DIALOG + GENERAL EVIDENCE GAP
            # ============================================================
            if round_display < agent_pipeline_start_round:
                # Diagnosis from dialog (for early stopping)
                diag_result = await diagnosis_dialog_agent.execute(context)
                if diag_result.success and diag_result.predictions:
                    final_diagnosis = diag_result.disease
                    final_confidence = diag_result.confidence
                    all_preds = [
                        {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                        for p in diag_result.predictions
                    ]
                    context.update_last_turn_differential_diagnosis(
                        disease=final_diagnosis, confidence=final_confidence, final_predictions=all_preds
                    )
                    context.update_last_turn_prediction_status("success")
                    print(f"[Diagnosis (log)] {final_diagnosis} ({final_confidence}%)")
                    _qp_predictions("DIAGNOSIS FROM DIALOG (diagnosis_dialog_agent)", all_preds)
                else:
                    error_log.append(f"Round {round_display}: diagnosis_dialog_agent failed")
                    completion_status = "incomplete"
                    break
                
                # General evidence gap (from early_eg_start_round - 1 so report is ready)
                if use_evidence_gap and round_display >= early_eg_start_round - 1:
                    use_dialog_for_eg = self.config.get("use_dialog_for_evidence_gap", False)
                    if use_dialog_for_eg:
                        gen_gap_result = await evidence_gap_agent.execute_general(
                            context, dialog=context.get_dialog_readable()
                        )
                    else:
                        gen_gap_result = await evidence_gap_agent.execute_general(
                            context, str_summary=para_summary or str_summary
                        )
                    if gen_gap_result.success:
                        general_gap_report = gen_gap_result.format_for_doctor()
                        print(f"  [GENERAL_GAP] {len(gen_gap_result.report.general_gaps)} gaps identified")
                        _general_str = "\n".join(gen_gap_result.report.general_gaps) if gen_gap_result.report.general_gaps else None
                        context.update_last_turn_evidence_gaps(general=_general_str)
                        if qualitative_print:
                            _qp_section("GENERAL EVIDENCE GAP (evidence_gap_agent.execute_general)")
                            if gen_gap_result.report.general_gaps:
                                for g in gen_gap_result.report.general_gaps:
                                    _qp(f"  - {g}")
                            else:
                                _qp("  (no gaps)")
                    else:
                        error_log.append(f"Round {round_display}: evidence_gap_agent (general) failed - {gen_gap_result.error}")
                
                # Check early exit
                self._print_turn_tokens(round_display)
                if final_confidence >= high_conf_threshold:
                    break
                
                continue
            
            # ============================================================
            # FULL PIPELINE ROUNDS (agent_pipeline_start_round+)
            # ============================================================
            
            # --- SPECIALIST ENSEMBLE: 10x diagnosis_summary_agent + ensemble merge ---
            if use_specialist_ensemble:
                specialist_preds: dict[str, list[dict]] = {}

                all_specs = [("General Physician", None)] + [(s, s) for s in SPECIALTIES]

                # SEQUENTIAL: Run one at a time
                for spec_name, specialty in all_specs:
                    spec_result = await specialist_diagnosis_agent.execute(context, specialty=specialty)
                    if spec_result.success and spec_result.predictions:
                        specialist_preds[spec_name] = [
                            {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                            for p in spec_result.predictions
                        ]
                        print(f"  [SPECIALIST] {spec_name}: {spec_result.disease} ({spec_result.confidence}%)")
                        _qp_predictions(f"SPECIALIST — {spec_name}", specialist_preds[spec_name])
                    else:
                        error_log.append(f"Round {round_display}: diagnosis_summary ({spec_name}) failed")

                # Ensemble merge
                if not specialist_preds:
                    error_log.append(f"Round {round_display}: all specialist diagnoses failed")
                    completion_status = "incomplete"
                    break
                
                # Log compact per-turn specialist predictions (disease + confidence only)
                _compact_specialist_preds = {
                    spec: [{"disease": p["disease"], "confidence": p["confidence"]} for p in preds]
                    for spec, preds in specialist_preds.items()
                }
                context.update_last_turn_specialist_predictions(
                    json.dumps(_compact_specialist_preds, separators=(",", ":"))
                )
                
                ensemble_result = await specialist_ensemble_agent.execute(
                    context,
                    specialist_predictions=specialist_preds,
                    dialog=context.get_dialog_readable()
                )
                if not ensemble_result.success or not ensemble_result.predictions:
                    error_log.append(f"Round {round_display}: specialist_ensemble_agent failed - {ensemble_result.error}")
                    completion_status = "incomplete"
                    break
                
                ensemble_predictions = [
                    {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                    for p in ensemble_result.predictions
                ]
                context.add_trace("specialist_ensemble_agent", "merge", {
                    "specialist_predictions": specialist_preds,
                    "ensemble_predictions": ensemble_predictions
                })
                print(f"  [ENSEMBLE] {ensemble_predictions[0]['disease']} ({ensemble_predictions[0]['confidence']}%)")
                _qp_predictions("ENSEMBLE MERGE (specialist_ensemble_agent)", ensemble_predictions)
                
                # Store as LLM predictions for logging
                final_llm_predictions = ensemble_predictions
            else:
                # --- SINGLE GENERAL PREDICTION (no specialist ensemble) ---
                print("SINGLE GENERAL PREDICTION, not ensembling specialists")
                general_result = await specialist_diagnosis_agent.execute(context, specialty=None)
                if not general_result.success or not general_result.predictions:
                    error_log.append(f"Round {round_display}: specialist_diagnosis_agent (general) failed - {general_result.error}")
                    completion_status = "incomplete"
                    break
                
                ensemble_predictions = [
                    {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                    for p in general_result.predictions
                ]
                context.add_trace("specialist_diagnosis_agent", "general_only", {
                    "predictions": ensemble_predictions
                })
                print(f"  [GENERAL] {ensemble_predictions[0]['disease']} ({ensemble_predictions[0]['confidence']}%)")
                _qp_predictions("GENERAL PREDICTION (specialist_diagnosis_agent)", ensemble_predictions)
                
                # Store as LLM predictions for logging
                final_llm_predictions = ensemble_predictions
            
            # --- KB PIPELINE ---
            kb_predictions = None
            if umls_client is not None and str_summary is not None:
                try:
                    kb_candidates, kb_candidates_formatted = get_kb_disease_candidates(
                        str_summary, umls_client, max_candidates=10000
                    )
                except Exception as e:
                    kb_candidates = []
                    error_log.append(f"Round {round_display}: KG lookup failed - {str(e)}")
                
                if kb_candidates:
                    if qualitative_print:
                        _qp_section(f"KB CANDIDATE DISEASES (UMLS lookup, {len(kb_candidates)} total)")
                        for c in kb_candidates:
                            disease_name = (c.get("disease_name") or c.get("disease") or c.get("name") or str(c)).title()
                            match_count = c.get("symptom_match_count")
                            matched = c.get("matched_symptoms") or []
                            if match_count is not None:
                                _qp(f"  - {disease_name}  [matches {match_count}: {', '.join(matched)}]")
                            else:
                                _qp(f"  - {disease_name}")
                    use_dialog_for_kb = self.config.get("use_dialog_for_kb", False)
                    if use_dialog_for_kb:
                        kb_result = await diagnosis_kb_agent.execute(
                            context, dialog=context.get_dialog_readable(), kb_candidates=kb_candidates_formatted
                        )
                    else:
                        kb_input = para_summary if para_summary else str_summary
                        kb_result = await diagnosis_kb_agent.execute(
                            context, str_summary=kb_input, kb_candidates=kb_candidates_formatted
                        )
                    if kb_result.success and kb_result.predictions:
                        kb_predictions = kb_result.predictions
                        final_kb_predictions = [{"disease": p.disease, "reasoning": p.reasoning} for p in kb_predictions]
                        print(f"  [KB] Top: {kb_predictions[0].disease}")
                        _qp_predictions("KB DIAGNOSIS (diagnosis_kb_agent)", final_kb_predictions)
                    else:
                        error_log.append(f"Round {round_display}: diagnosis_kb_agent failed")
                else:
                    print(f"  [KB] No candidates from UMLS, skipping KB diagnosis")
                    _qp_section("KB CANDIDATE DISEASES")
                    _qp("  (no candidates returned by UMLS)")
            
            # --- MERGE: ensemble 3 + KB 3 → final 3 ---
            if kb_predictions:
                kb_preds_dicts = [{"disease": p.disease, "reasoning": p.reasoning} for p in kb_predictions]
                llm_preds_dicts = [{"disease": p["disease"], "reasoning": p.get("reasoning", "")} for p in ensemble_predictions]
                
                use_dialog_for_merge = self.config.get("use_dialog_for_merge", False)
                if use_dialog_for_merge:
                    merge_result = await diagnosis_merge_agent.execute(
                        context,
                        dialog=context.get_dialog_readable(),
                        kb_predictions=kb_preds_dicts,
                        llm_predictions=llm_preds_dicts
                    )
                else:
                    merge_result = await diagnosis_merge_agent.execute(
                        context,
                        str_summary=para_summary or str_summary,
                        kb_predictions=kb_preds_dicts,
                        llm_predictions=llm_preds_dicts
                    )
                if merge_result.success and merge_result.predictions:
                    final_predictions = merge_result.predictions
                    final_diagnosis = final_predictions[0].disease
                    final_confidence = final_predictions[0].confidence
                    all_predictions_formatted = [
                        {"disease": p.disease, "confidence": p.confidence, "reasoning": p.reasoning}
                        for p in final_predictions
                    ]
                    current_predictions = all_predictions_formatted
                    context.add_trace("diagnosis_merge_agent", "merge", {"predictions": all_predictions_formatted})
                    print(f"  [MERGE] {final_diagnosis} ({final_confidence}%)")
                    _qp_predictions("FINAL MERGE (diagnosis_merge_agent: ensemble + KB)", all_predictions_formatted)
                else:
                    error_log.append(f"Round {round_display}: diagnosis_merge_agent failed")
                    # Use ensemble predictions directly
                    final_diagnosis = ensemble_predictions[0]["disease"]
                    final_confidence = ensemble_predictions[0]["confidence"]
                    all_predictions_formatted = ensemble_predictions
                    current_predictions = all_predictions_formatted
            else:
                # No KB → use ensemble predictions directly
                final_diagnosis = ensemble_predictions[0]["disease"]
                final_confidence = ensemble_predictions[0]["confidence"]
                all_predictions_formatted = ensemble_predictions
                current_predictions = all_predictions_formatted
            
            # Log predictions
            llm_preds_str = json.dumps(final_llm_predictions, separators=(',', ':')) if final_llm_predictions else None
            kb_preds_str = json.dumps(final_kb_predictions, separators=(',', ':')) if final_kb_predictions else None
            
            context.update_last_turn_differential_diagnosis(
                disease=final_diagnosis,
                confidence=final_confidence,
                final_predictions=all_predictions_formatted,
                llm_predictions=llm_preds_str,
                kb_predictions=kb_preds_str
            )
            context.update_last_turn_prediction_status("success")
            print(f"[Diagnosis] {final_diagnosis} ({final_confidence}%)")
            if qualitative_print:
                _qp_section("FINAL DIAGNOSIS (this round)")
                _qp(f"  {final_diagnosis}  ({final_confidence}%)")
            
            # --- EVIDENCE GAP (differential + general) ---
            if use_evidence_gap:
                use_dialog_for_eg = self.config.get("use_dialog_for_evidence_gap", False)
                if use_dialog_for_eg:
                    gap_result = await evidence_gap_agent.execute(
                        context,
                        predictions=all_predictions_formatted,
                        dialog=context.get_dialog_readable()
                    )
                else:
                    gap_result = await evidence_gap_agent.execute(
                        context,
                        predictions=all_predictions_formatted,
                        str_summary=para_summary or str_summary
                    )
                if gap_result.success:
                    evidence_gap_report = gap_result.format_for_doctor()
                    print(f"  [EVIDENCE_GAP] {len(gap_result.report.diagnosis_specific_gaps)} diagnosis-specific gaps")
                    _general_str = "\n".join(gap_result.report.general_gaps) if gap_result.report.general_gaps else None
                    _diff_str = "\n".join(
                        f"{d.disease}: {d.missing_evidence}" for d in gap_result.report.diagnosis_specific_gaps
                    ) if gap_result.report.diagnosis_specific_gaps else None
                    context.update_last_turn_evidence_gaps(general=_general_str, differential=_diff_str)
                    if qualitative_print:
                        _qp_section("EVIDENCE GAP REPORT (evidence_gap_agent)")
                        if gap_result.report.general_gaps:
                            _qp("  General gaps:")
                            for g in gap_result.report.general_gaps:
                                _qp(f"    - {g}")
                        if gap_result.report.diagnosis_specific_gaps:
                            _qp("  Diagnosis-specific gaps:")
                            for diag in gap_result.report.diagnosis_specific_gaps:
                                _qp(f"    {diag.disease}:")
                                _qp(f"      {diag.missing_evidence}")
                        if not gap_result.report.general_gaps and not gap_result.report.diagnosis_specific_gaps:
                            _qp("  (no gaps identified)")
                else:
                    error_log.append(f"Round {round_display}: evidence_gap_agent failed - {gap_result.error}")
            
            # --- EXIT CHECK ---
            self._print_turn_tokens(round_display)
            if final_confidence >= high_conf_threshold:
                print(f"  [EXIT] Confidence {final_confidence}% >= {high_conf_threshold}%")
                break
        
        # ============================================================
        # JUDGE
        # ============================================================
        if final_diagnosis:
            context.set("predicted_disease", final_diagnosis)
            judge_result = await judge_agent.execute(context)
            is_correct = judge_result.same if judge_result.success else False
            judge_explanation = judge_result.explanation if judge_result.success else None
            if not judge_result.success:
                error_log.append(f"judge_agent failed - {judge_result.error}")
        else:
            is_correct = False
            judge_explanation = None
            final_diagnosis = "No diagnosis"
        
        # Print judge tokens
        j_in, j_out = judge_agent.get_turn_tokens()
        # if j_in > 0 or j_out > 0:
        #     print(f"  [TOKENS] Judge — Input: {j_in:,} | Output: {j_out:,} | Total: {j_in + j_out:,}")
        
        llm_preds_str = json.dumps(final_llm_predictions, separators=(',', ':')) if final_llm_predictions else None
        kb_preds_str = json.dumps(final_kb_predictions, separators=(',', ':')) if final_kb_predictions else None
        
        return CaseResult(
            case_id=context.get("case_id"),
            ground_truth=context.get("ground_truth"),
            predicted_disease=final_diagnosis,
            confidence=final_confidence,
            result="correct" if is_correct else "incorrect",
            judge_explanation=judge_explanation,
            rounds=context.get_round_count(),
            completion_status=completion_status,
            error_log=error_log if error_log else None,
            dialog_history=context.get_dialog_structured(),
            summary=final_summary,
            llm_predictions=llm_preds_str,
            kb_predictions=kb_preds_str
        )


