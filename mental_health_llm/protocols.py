"""
Protocol interfaces for mental health LLM components.

Defines structural typing contracts using ``typing.Protocol`` so that
components can be swapped without inheritance.  Existing classes
(KeywordRouter, EmbeddingRouter, CrisisGate, etc.) already satisfy
these protocols structurally — no code changes needed.

Also includes lightweight mock implementations for unit testing.

Source pattern: claw-code trait-based interfaces (ApiClient, ToolExecutor,
PermissionPrompter).

Usage:

    from mental_health_llm.protocols import BaseRouter, BaseSafetyGate

    def process(router: BaseRouter, gate: BaseSafetyGate) -> str:
        if gate.check("I want to end it all")["is_crisis"]:
            return router.get_system_prompt("crisis-intervention")
        return router.route("hello")

    # Works with any conforming implementation:
    process(KeywordRouter(), CrisisGate())
    process(MockRouter(), MockSafetyGate())
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Router protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseRouter(Protocol):
    """Structural interface for skill routers.

    Satisfied by: KeywordRouter, EmbeddingRouter, SkillRouter.
    """

    def route(self, message: str, history: Optional[list] = None) -> str:
        """Route a user message to the best-matching skill name."""
        ...

    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        """Route with confidence score and details dict."""
        ...

    def get_system_prompt(self, skill_name: str) -> str:
        """Return the system prompt for a skill."""
        ...

    def list_skills(self) -> list:
        """Return skill names in priority order."""
        ...


# ---------------------------------------------------------------------------
# Safety gate protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseSafetyGate(Protocol):
    """Structural interface for pre-routing safety gates.

    Satisfied by: CrisisGate.
    """

    def check(self, message: str, history: Optional[list] = None) -> dict:
        """Check whether a message indicates crisis.

        Returns a dict with at least ``is_crisis`` (bool).
        """
        ...


# ---------------------------------------------------------------------------
# Generator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseGenerator(Protocol):
    """Structural interface for text generators.

    Allows swapping local LoRA models with API-based models (GPT-4, Claude).
    """

    def generate(
        self,
        question: str,
        system_prompt: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
        max_length: int = 1024,
    ) -> str:
        """Generate a counseling response."""
        ...


# ---------------------------------------------------------------------------
# Session store protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseSessionStore(Protocol):
    """Structural interface for session persistence backends.

    Satisfied by: SQLiteSessionStore.
    """

    def save_turn(
        self,
        user_id: int,
        user_msg: str,
        assistant_msg: str,
        **kwargs: Any,
    ) -> None:
        """Persist a conversation turn."""
        ...

    def load_session(self, user_id: int) -> Optional[dict[str, Any]]:
        """Load a user's session data, or None if expired/missing."""
        ...

    def delete_session(self, user_id: int) -> None:
        """Delete a user's session."""
        ...


# ---------------------------------------------------------------------------
# Response guard protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseResponseGuard(Protocol):
    """Structural interface for post-generation safety validation.

    Satisfied by: ResponseGuard.
    """

    def validate(
        self, response: str, skill: str, crisis_level: str = "none"
    ) -> Any:
        """Validate a generated response. Returns a result with .response and .action."""
        ...


# =========================================================================
# Mock implementations (for testing)
# =========================================================================


class MockRouter:
    """Test double that always routes to a configurable skill."""

    def __init__(self, default_skill: str = "general-support") -> None:
        self._skill = default_skill
        self._prompts: dict[str, str] = {
            "crisis-intervention": "You are a crisis counselor.",
            "general-support": "You are a supportive counselor.",
            "cbt-therapy": "You are a CBT therapist.",
        }

    def route(self, message: str, history: Optional[list] = None) -> str:
        return self._skill

    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        return self._skill, 1.0, {"router_type": "mock", "scores": {}}

    def get_system_prompt(self, skill_name: str) -> str:
        return self._prompts.get(skill_name, "You are a counselor.")

    def get_adapter_path(self, skill_name: str) -> str:
        return ""

    def list_skills(self) -> list:
        return list(self._prompts.keys())


class MockSafetyGate:
    """Test double that optionally flags every message as crisis."""

    def __init__(self, *, always_crisis: bool = False) -> None:
        self._always_crisis = always_crisis

    def check(self, message: str, history: Optional[list] = None) -> dict:
        return {
            "is_crisis": self._always_crisis,
            "keyword_triggered": False,
            "embedding_triggered": False,
            "embedding_score": 0.0,
        }


class MockGenerator:
    """Test double that echoes the question back."""

    def __init__(self, response: str = "I hear you.") -> None:
        self._response = response

    def generate(
        self,
        question: str,
        system_prompt: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
        max_length: int = 1024,
    ) -> str:
        return self._response


class MockSessionStore:
    """In-memory session store for testing."""

    def __init__(self) -> None:
        self._sessions: dict[int, dict] = {}

    def save_turn(
        self,
        user_id: int,
        user_msg: str,
        assistant_msg: str,
        **kwargs: Any,
    ) -> None:
        if user_id not in self._sessions:
            self._sessions[user_id] = {
                "user_id": user_id,
                "messages": [],
                "crisis_flags": [],
                "active_skill": "",
                "model_key": "",
            }
        self._sessions[user_id]["messages"].append([user_msg, assistant_msg])
        if kwargs.get("is_crisis"):
            idx = len(self._sessions[user_id]["messages"]) - 1
            self._sessions[user_id]["crisis_flags"].append(idx)
        if kwargs.get("skill"):
            self._sessions[user_id]["active_skill"] = kwargs["skill"]

    def load_session(self, user_id: int) -> Optional[dict[str, Any]]:
        return self._sessions.get(user_id)

    def delete_session(self, user_id: int) -> None:
        self._sessions.pop(user_id, None)
