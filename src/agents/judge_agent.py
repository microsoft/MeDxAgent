"""Judge Agent - Compares predicted disease with ground truth."""

from src.agents.base_agent import BaseAgent, AgentOutput
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.workflow_context import WorkflowContext


class JudgmentResponse(BaseModel):
    """Structured response for disease comparison judgment."""
    same: bool = Field(..., description="true if the conditions are essentially the same disease OR if the predicted disease is a specific type/subtype of the ground truth, false if they are different conditions")
    explanation: str = Field(default="", description="Brief explanation of the comparison")


class JudgeAgentOutput(AgentOutput):
    """Output from the judge agent."""
    same: bool = Field(default=False, description="true if conditions are the same or predicted is a subtype of ground truth, false otherwise")
    explanation: str = Field(default="", description="Brief explanation of the comparison")
    matched_gt: str | None = Field(default=None, description="The ground truth disease that was matched (if any)")


class JudgeAgent(BaseAgent):
    """Agent that judges if two disease names refer to the same condition.
    
    The judge agent:
    - Receives predicted disease and ground truth disease(s)
    - Determines if prediction matches any of the ground truth diseases
    - Handles variations in naming (e.g., "Type 2 Diabetes" vs "Diabetes Mellitus Type 2")
    - Returns same/not same with explanation
    """
    
    agent_name = "judge.default"

    async def execute(
        self,
        context: "WorkflowContext",
        disease: str | None = None,
        gt: str | list[str] | None = None,
        **kwargs
    ) -> JudgeAgentOutput:
        """Judge if predicted and ground truth diseases are the same.
        
        Args:
            context: Workflow context
            disease: The predicted disease
            gt: The ground truth disease(s) - can be a string or list of strings
            
        Returns:
            JudgeAgentOutput with same status and explanation
        """
        # Get from parameters or context
        predicted = disease or context.get("predicted_disease", "")
        ground_truth = gt or context.get("ground_truth", "")
        
        if not predicted:
            return JudgeAgentOutput(
                success=False,
                error="No predicted disease provided",
                same=False,
                explanation=""
            )
        
        if not ground_truth:
            return JudgeAgentOutput(
                success=False,
                error="No ground truth disease provided",
                same=False,
                explanation=""
            )
        
        # Normalize ground_truth to a list
        if isinstance(ground_truth, str):
            gt_list = [ground_truth]
        else:
            gt_list = ground_truth
        
        # Check against each ground truth disease
        for gt_disease in gt_list:
            user_prompt = self.get_user_prompt(
                "user_prompt_template",
                predicted=predicted,
                ground_truth=gt_disease
            )

            try:
                result = await self._call_llm_structured(user_prompt, JudgmentResponse)
                
                if result.same:
                    # Found a match - return success
                    return JudgeAgentOutput(
                        success=True,
                        same=True,
                        explanation=result.explanation,
                        matched_gt=gt_disease
                    )
            except Exception as e:
                # Continue to next GT on error
                print(f"  [LLM_ERROR] judge_agent failed: {str(e)}")
                continue
        
        # No match found with any GT
        # Return the explanation from the last comparison
        try:
            # Use the last GT for the final explanation
            user_prompt = self.get_user_prompt(
                "user_prompt_template",
                predicted=predicted,
                ground_truth=gt_list[-1] if gt_list else ""
            )
            result = await self._call_llm_structured(user_prompt, JudgmentResponse)
            explanation = result.explanation
        except Exception:
            explanation = f"Prediction '{predicted}' did not match any of the ground truth diseases: {gt_list}"
        
        return JudgeAgentOutput(
            success=True,
            same=False,
            explanation=explanation
        )
