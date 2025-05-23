"""Define the configurable parameters for the agent."""
from __future__ import annotations
import os

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    reasoner_prompt: str = field(
        default=prompts.REASONER_PROMPT,
        metadata={
            "description": "The system prompt for the reasoning/research phase."
        },
    )

    final_response_prompt: str = field(
        default=prompts.FINAL_RESPONSE_PROMPT,
        metadata={
            "description": "The system prompt for the final response phase."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="mistralai/mistral-small-latest",
        # default="anthropic/claude-3-5-haiku-latest",
        # default="anthropic/claude-3-5-sonnet-latest",
        # default="anthropic/claude-3-7-sonnet-latest",
        default="openai/gpt-4.1",
        # default="ollama/phi4-mini",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Meteor configuration
    meteor_url: str = os.environ.get("DEFAULT_METEOR_URL") or "ws://127.0.0.1:3000/websocket"
    meteor_user_name: str = os.environ.get("DEFAULT_METEOR_USERNAME") or "fnord"
    meteor_user_password: str = os.environ.get("DEFAULT_METEOR_PASSWORD") or "fnord"
    
    meteor_prefix: str = field(
        default='DEFAULT',
        metadata={
            "description": "The prefix for the env variables for the Meteor connection. "
            "This is used to differentiate between different Meteor clients."
        },
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
