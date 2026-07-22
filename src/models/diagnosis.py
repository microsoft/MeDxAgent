"""Diagnosis result data models."""

from pydantic import BaseModel, Field


class DiagnosisResult(BaseModel):
    """Result from a diagnosis agent."""
    
    disease: str = Field(..., description="Predicted disease name")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    reasoning: str | None = Field(None, description="Brief reasoning for the diagnosis")


class CaseResult(BaseModel):
    """Final result for a single patient case evaluation."""
    
    case_id: str = Field(..., description="The case identifier")
    ground_truth: str | list[str] = Field(..., description="Ground truth disease (single or multiple)")
    predicted_disease: str = Field(..., description="Predicted disease by the system")
    confidence: int = Field(..., description="Final confidence score")
    result: str = Field(..., description="'correct' or 'incorrect'")
    rounds: int = Field(..., description="Number of dialog rounds")
    completion_status: str = Field(default="complete", description="'complete' if finished normally, 'incomplete' if ended due to error")
    error_log: list[str] | None = Field(default=None, description="List of errors encountered during execution")
    
    # Trace information (optional, for debugging)
    dialog_history: list[dict] | None = Field(None, description="Structured dialog history with rounds")
    summary: str | None = Field(None, description="Generated summary if applicable")
    execution_time_seconds: float | None = Field(None, description="Time taken to execute this case in seconds")
    trace: list[dict] | None = Field(None, description="Execution trace for debugging agent failures")
    
    # Diagnosis source predictions (for KG-integrated workflows)
    llm_predictions: str | None = Field(None, description="LLM diagnosis predictions as compact JSON string")
    kb_predictions: str | None = Field(None, description="KB diagnosis predictions as compact JSON string")
    
    # Judge reasoning
    judge_explanation: str | None = Field(None, description="Judge agent's explanation for the correctness decision")


class EvaluationSummary(BaseModel):
    """Summary of evaluation across all cases."""
    
    total_cases: int = Field(..., description="Total number of cases evaluated")
    correct: int = Field(..., description="Number of correct diagnoses")
    incorrect: int = Field(..., description="Number of incorrect diagnoses")
    accuracy: float = Field(..., description="Accuracy percentage")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_rounds: float = Field(..., description="Average dialog rounds")
    results: list[CaseResult] = Field(..., description="Individual case results")
