from __future__ import annotations

from collections import Counter
from typing import Any

from .base import BaseAgent, ChallengeDecision, PlayDecision


class InvalidAgentError(Exception):
    """Raised when a player agent cannot safely interact with the game."""


class LLMAPIError(RuntimeError):
    """Raised for likely LLM API, auth, or model-proxy failures."""


def is_raw_kbench_llm(value: Any) -> bool:
    if value is None:
        return False
    if callable(getattr(value, "bind", None)):
        return False
    return callable(getattr(value, "prompt", None)) and callable(
        getattr(value, "respond", None)
    )


def validate_agent_shape(agent: Any, *, player_name: str) -> None:
    if agent is None:
        raise InvalidAgentError(
            f"{player_name} has no agent. player_configs entries must include "
            "{'agent': ...}."
        )
    for method in ("bind", "choose_play", "choose_challenge"):
        if not callable(getattr(agent, method, None)):
            raise InvalidAgentError(
                f"{player_name} agent {type(agent).__name__} must define callable "
                f"{method}()."
            )
    if getattr(type(agent), "choose_play", None) is BaseAgent.choose_play:
        raise InvalidAgentError(
            f"{player_name} agent {type(agent).__name__} must implement choose_play()."
        )
    if getattr(type(agent), "choose_challenge", None) is BaseAgent.choose_challenge:
        raise InvalidAgentError(
            f"{player_name} agent {type(agent).__name__} must implement "
            "choose_challenge()."
        )


def bind_and_validate_agent(agent: Any, participant, game):
    validate_agent_shape(agent, player_name=participant.name)
    try:
        bound_agent = agent.bind(participant, game)
    except Exception as exc:
        raise InvalidAgentError(
            f"{participant.name} agent {type(agent).__name__}.bind() failed: {exc}"
        ) from exc
    if bound_agent is None:
        raise InvalidAgentError(
            f"{participant.name} agent {type(agent).__name__}.bind() returned None."
        )
    validate_agent_shape(bound_agent, player_name=participant.name)
    if getattr(bound_agent, "participant", None) is not participant:
        raise InvalidAgentError(
            f"{participant.name} agent {type(bound_agent).__name__} must set "
            "self.participant to the bound participant."
        )
    if callable(getattr(bound_agent, "setup", None)):
        try:
            bound_agent.setup()
        except Exception as exc:
            raise InvalidAgentError(
                f"{participant.name} agent {type(bound_agent).__name__}.setup() "
                f"failed: {exc}"
            ) from exc
    return bound_agent


def coerce_play_decision(value: Any) -> PlayDecision:
    if isinstance(value, PlayDecision):
        return value
    if isinstance(value, dict):
        return PlayDecision(
            played_cards=list(value.get("played_cards", [])),
            behavior=str(value.get("behavior", "")),
            play_reason=str(value.get("play_reason", "")),
        )
    played_cards = getattr(value, "played_cards", None)
    behavior = getattr(value, "behavior", None)
    play_reason = getattr(value, "play_reason", None)
    if played_cards is not None and behavior is not None and play_reason is not None:
        return PlayDecision(
            played_cards=list(played_cards),
            behavior=str(behavior),
            play_reason=str(play_reason),
        )
    raise InvalidAgentError(f"Invalid play decision object: {value!r}")


def validate_play_decision(
    value: Any, hand: list[str], *, allow_empty_behavior: bool = False
) -> PlayDecision:
    decision = coerce_play_decision(value)
    if not isinstance(decision.played_cards, list):
        raise InvalidAgentError("played_cards must be a list.")
    if not 1 <= len(decision.played_cards) <= 3:
        raise InvalidAgentError("played_cards must contain 1 to 3 cards.")
    if not allow_empty_behavior and not str(decision.behavior).strip():
        raise InvalidAgentError("behavior must be non-empty public communication.")
    if not str(decision.play_reason).strip():
        raise InvalidAgentError("play_reason must be non-empty private reasoning.")
    hand_counts = Counter(hand)
    played_counts = Counter(decision.played_cards)
    illegal = {
        card: count
        for card, count in played_counts.items()
        if hand_counts[card] < count
    }
    if illegal:
        raise InvalidAgentError(
            f"played_cards must be selected from current hand. Illegal cards: {illegal}"
        )
    return decision


def coerce_challenge_decision(value: Any) -> ChallengeDecision:
    if isinstance(value, ChallengeDecision):
        return value
    if isinstance(value, dict):
        return ChallengeDecision(
            was_challenged=value.get("was_challenged"),
            challenge_reason=str(value.get("challenge_reason", "")),
        )
    was_challenged = getattr(value, "was_challenged", None)
    challenge_reason = getattr(value, "challenge_reason", None)
    if was_challenged is not None and challenge_reason is not None:
        return ChallengeDecision(
            was_challenged=was_challenged,
            challenge_reason=str(challenge_reason),
        )
    raise InvalidAgentError(f"Invalid challenge decision object: {value!r}")


def validate_challenge_decision(
    value: Any, *, allow_empty_reason: bool = False
) -> ChallengeDecision:
    decision = coerce_challenge_decision(value)
    if not isinstance(decision.was_challenged, bool):
        raise InvalidAgentError("was_challenged must be a boolean.")
    if not allow_empty_reason and not str(decision.challenge_reason).strip():
        raise InvalidAgentError("challenge_reason must be non-empty.")
    return decision


def likely_api_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "api key",
        "auth",
        "credential",
        "model_proxy",
        "model proxy",
        "rate limit",
        "quota",
        "timeout",
        "timed out",
        "connection",
        "connect",
        "ssl",
        "http 401",
        "http 403",
        "http 429",
        "401",
        "403",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in markers)
