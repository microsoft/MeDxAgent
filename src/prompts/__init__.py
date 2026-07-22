"""Prompt loader utility for loading prompts from JSON file."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class PromptLoader:
    """Loads and manages prompts from JSON configuration file."""
    
    def __init__(self, prompts_path: str | Path | None = None):
        """Initialize the prompt loader.
        
        Args:
            prompts_path: Path to prompts JSON file. Defaults to configs/prompts/prompts.json
        """
        if prompts_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            prompts_path = project_root / "configs" / "prompts" / "prompts.json"
        
        self.prompts_path = Path(prompts_path)
        self._prompts: dict[str, Any] = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompts from JSON file."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")
        
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            self._prompts = json.load(f)
    
    def reload(self) -> None:
        """Reload prompts from file (useful for hot-reloading during development)."""
        self._load_prompts()
    
    def _resolve(self, agent_name: str) -> dict:
        """Resolve a dot-notation agent name (e.g., 'diagnosis.summary') to its prompt dict.
        
        Args:
            agent_name: Dot-notation key like 'diagnosis.summary' or 'patient.default'
            
        Returns:
            The prompt dictionary for that agent
        """
        parts = agent_name.split(".")
        current = self._prompts
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"No prompts found for agent: {agent_name}")
            current = current[part]
        if not isinstance(current, dict):
            raise KeyError(f"No prompts found for agent: {agent_name}")
        return current

    def get_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent.
        
        Args:
            agent_name: Dot-notation name of the agent (e.g., 'patient.default')
            
        Returns:
            The system prompt string
        """
        return self._resolve(agent_name).get("system_prompt", "")
    
    def get_user_prompt_template(self, agent_name: str, template_name: str = "user_prompt_template") -> str:
        """Get a user prompt template for an agent.
        
        Args:
            agent_name: Dot-notation name of the agent
            template_name: Name of the template (default: 'user_prompt_template')
            
        Returns:
            The prompt template string
        """
        return self._resolve(agent_name).get(template_name, "")
    
    def format_user_prompt(self, agent_name: str, template_name: str = "user_prompt_template", **kwargs) -> str:
        """Get and format a user prompt template with provided variables.
        
        Args:
            agent_name: Dot-notation name of the agent
            template_name: Name of the template
            **kwargs: Variables to format into the template
            
        Returns:
            The formatted prompt string
        """
        template = self.get_user_prompt_template(agent_name, template_name)
        return template.format(**kwargs)
    
    def get_agent_prompts(self, agent_name: str) -> dict[str, str]:
        """Get all prompts for an agent.
        
        Args:
            agent_name: Dot-notation name of the agent
            
        Returns:
            Dictionary of all prompts for the agent
        """
        return self._resolve(agent_name).copy()


# Global cached instance
_prompt_loader: PromptLoader | None = None


def get_prompt_loader(prompts_path: str | Path | None = None) -> PromptLoader:
    """Get the global prompt loader instance.
    
    Args:
        prompts_path: Optional path to prompts file (only used on first call)
        
    Returns:
        The PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_path)
    return _prompt_loader


def reload_prompts() -> None:
    """Reload prompts from file (useful for development)."""
    global _prompt_loader
    if _prompt_loader is not None:
        _prompt_loader.reload()
