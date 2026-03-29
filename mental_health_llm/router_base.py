"""
Abstract base class for skill routers.

All router implementations (keyword, embedding, etc.) inherit from BaseRouter
so that consumers (skill_inference.py, telegram_bot.py) can use any backend
through a uniform interface.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseRouter(ABC):
    """Abstract base class for skill routers."""

    @abstractmethod
    def route(self, message: str, history: Optional[list] = None) -> str:
        """
        Route a user message to the best-matching skill.

        Args:
            message: The user's input message.
            history: Optional list of (user_msg, assistant_msg) tuples
                     for conversation-aware routing. Ignored by some backends.

        Returns:
            Skill name string (e.g. "crisis-intervention").
        """

    @abstractmethod
    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        """
        Route a message and return confidence details.

        Args:
            message: The user's input message.
            history: Optional conversation history.

        Returns:
            Tuple of (skill_name, confidence, details_dict).
        """

    @abstractmethod
    def get_system_prompt(self, skill_name: str) -> str:
        """Get the system prompt for a given skill."""

    @abstractmethod
    def get_adapter_path(self, skill_name: str) -> str:
        """Get the adapter path for a given skill."""

    @abstractmethod
    def list_skills(self) -> list:
        """Return list of all skill names in priority order."""
