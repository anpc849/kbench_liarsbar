from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .base import AgentContext, BaseAgent
from .validation import (
    InvalidAgentError,
    LLMAPIError,
    likely_api_failure,
    validate_challenge_decision,
    validate_play_decision,
)


@dataclass
class LLMPlayDecision:
    played_cards: list[str]
    behavior: str
    play_reason: str


@dataclass
class LLMChallengeDecision:
    was_challenged: bool
    challenge_reason: str


class DefaultLLMAgent(BaseAgent):
    """Default adapter for raw Kaggle Benchmarks LLM objects.

    This class validates and retries invalid model outputs. It never substitutes
    a strategic fallback action because that would corrupt behavior analysis.
    """

    def __init__(self, llm, max_retries: int = 5, llm_pause_seconds: float = 1.0):
        self.llm = llm
        self.max_retries = max_retries
        self.llm_pause_seconds = llm_pause_seconds
        self.opinions: dict[str, str] = {}
        self.decision_log: list[dict[str, Any]] = []

    def choose_play(self, context: AgentContext):
        retry_note = ""
        last_error = None
        last_raw = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._prompt_llm(
                    context=context,
                    phase="play",
                    attempt=attempt,
                    prompt=self._play_prompt(context, retry_note),
                    schema=LLMPlayDecision,
                )
                self._sleep_after_llm_call()
                last_raw = raw
                decision = validate_play_decision(raw, context.hand)
                self._record(
                    phase="play",
                    context=context,
                    requested=raw,
                    chosen=decision,
                    error=None,
                    attempt=attempt,
                )
                return decision
            except Exception as exc:
                if likely_api_failure(exc):
                    raise LLMAPIError(
                        f"{context.player_name} LLM API call failed during play: {exc}"
                    ) from exc
                last_error = exc
                retry_note = self._retry_note(str(exc), context)
        self._record(
            phase="play",
            context=context,
            requested=last_raw,
            chosen=None,
            error=str(last_error),
            attempt=self.max_retries,
        )
        raise InvalidAgentError(
            f"{context.player_name} could not produce a legal play after "
            f"{self.max_retries + 1} attempts. Last error: {last_error}"
        ) from last_error

    def choose_challenge(self, context: AgentContext):
        retry_note = ""
        last_error = None
        last_raw = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._prompt_llm(
                    context=context,
                    phase="challenge",
                    attempt=attempt,
                    prompt=self._challenge_prompt(context, retry_note),
                    schema=LLMChallengeDecision,
                )
                self._sleep_after_llm_call()
                last_raw = raw
                decision = validate_challenge_decision(raw)
                self._record(
                    phase="challenge",
                    context=context,
                    requested=raw,
                    chosen=decision,
                    error=None,
                    attempt=attempt,
                )
                return decision
            except Exception as exc:
                if likely_api_failure(exc):
                    raise LLMAPIError(
                        f"{context.player_name} LLM API call failed during "
                        f"challenge: {exc}"
                    ) from exc
                last_error = exc
                retry_note = self._retry_note(str(exc), context)
        self._record(
            phase="challenge",
            context=context,
            requested=last_raw,
            chosen=None,
            error=str(last_error),
            attempt=self.max_retries,
        )
        raise InvalidAgentError(
            f"{context.player_name} could not produce a legal challenge decision "
            f"after {self.max_retries + 1} attempts. Last error: {last_error}"
        ) from last_error

    def reflect(self, context: AgentContext):
        retry_note = ""
        last_error = None
        last_raw = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._prompt_llm(
                    context=context,
                    phase="reflection",
                    attempt=attempt,
                    prompt=self._reflection_prompt(context, retry_note),
                    schema=str,
                )
                self._sleep_after_llm_call()
                last_raw = raw
                opinions = self._parse_reflection(raw, context)
                self.opinions.update(opinions)
                self._record(
                    phase="reflect",
                    context=context,
                    requested=raw,
                    chosen={"opinions": opinions},
                    error=None,
                    attempt=attempt,
                )
                return dict(self.opinions)
            except Exception as exc:
                if likely_api_failure(exc):
                    raise LLMAPIError(
                        f"{context.player_name} LLM API call failed during "
                        f"reflection: {exc}"
                    ) from exc
                last_error = exc
                retry_note = self._reflection_retry_note(str(exc), context)
        self._record(
            phase="reflect",
            context=context,
            requested=last_raw,
            chosen=None,
            error=str(last_error),
            attempt=self.max_retries,
        )
        raise InvalidAgentError(
            f"{context.player_name} could not produce valid reflection memory "
            f"after {self.max_retries + 1} attempts. Last error: {last_error}"
        ) from last_error

    def _prompt_llm(
        self,
        *,
        context: AgentContext,
        phase: str,
        attempt: int,
        prompt: str,
        schema: type,
    ):
        chat_name = f"liarsbar-{context.player_name}-{phase}-attempt-{attempt + 1}"
        try:
            from kaggle_benchmarks import chats
        except ImportError:
            return self.llm.prompt(prompt, schema=schema)

        with chats.new(name=chat_name, orphan=False):
            return self.llm.prompt(prompt, schema=schema)

    @staticmethod
    def _parse_reflection(raw: Any, context: AgentContext) -> dict[str, str]:
        text = str(raw).strip()
        match = re.search(r"({[\s\S]*})", text)
        if not match:
            raise InvalidAgentError(
                f"reflection response must contain a JSON object. "
                f"Previous response: {text!r}"
            )
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise InvalidAgentError(f"reflection JSON could not be parsed: {exc}") from exc
        opinions = payload.get("opinions")
        if not isinstance(opinions, dict):
            raise InvalidAgentError("reflection JSON must contain an opinions object.")
        valid_names = set(context.alive_players)
        valid_names.discard(context.player_name)
        parsed = {}
        for name, opinion in opinions.items():
            name = str(name)
            if name == context.player_name:
                continue
            if valid_names and name not in valid_names:
                raise InvalidAgentError(
                    f"reflection included unknown or non-visible player {name!r}."
                )
            text_opinion = str(opinion).strip()
            if not text_opinion:
                raise InvalidAgentError(
                    f"reflection opinion for {name!r} must be non-empty."
                )
            parsed[name] = text_opinion
        if not parsed and valid_names:
            raise InvalidAgentError(
                "reflection opinions must include at least one other alive player."
            )
        return parsed

    def get_log(self) -> dict[str, Any] | None:
        return {"opinions": dict(self.opinions), "decisions": list(self.decision_log)}

    def model_name(self) -> str:
        for attr in ("model", "name", "id"):
            value = getattr(self.llm, attr, None)
            if value:
                return str(value)
        return type(self.llm).__name__

    def _play_prompt(self, context: AgentContext, retry_note: str) -> str:
        return (
            self._rules()
            + "\n\n"
            + context.to_text()
            + "\n\nIt is your turn to play cards. Choose 1 to 3 cards from your "
            "current hand. Other players will only see how many cards you claim "
            "to play and your public behavior, not the actual cards unless you "
            "are challenged.\n\nReturn JSON with fields: played_cards "
            "(list of card strings), behavior (public table behavior/utterance), "
            "and play_reason (private reasoning)."
            + retry_note
        )

    def _challenge_prompt(self, context: AgentContext, retry_note: str) -> str:
        return (
            self._rules()
            + "\n\n"
            + context.to_text()
            + "\n\nDecide whether to challenge the previous player's claim. "
            "Return JSON with fields: was_challenged (boolean) and "
            "challenge_reason (private reasoning)."
            + retry_note
        )

    def _reflection_prompt(self, context: AgentContext, retry_note: str) -> str:
        other_players = [name for name in context.alive_players if name != context.player_name]
        return (
            "You are updating private opponent impressions for future Liars Bar "
            "rounds. Use only the public history, revealed information, your "
            "private decision history, and your private opinions shown below. "
            "Do not invent hidden hands or bullet positions.\n\n"
            f"{context.to_text()}\n\n"
            "Return only a JSON object, no markdown, with this shape:\n"
            '{"opinions": {"Opponent Name": "brief strategy impression"}}\n'
            f"Include exactly these other alive players when possible: "
            f"{', '.join(other_players) if other_players else 'none'}."
            + retry_note
        )

    @staticmethod
    def _rules() -> str:
        return (
            "Liars Bar rules: each round has target card Q, K, or A. Players "
            "claim every played card is the target card, but may bluff. Jokers "
            "count as valid target cards. A challenge succeeds if any revealed "
            "played card is neither the target card nor Joker. The loser fires "
            "their own revolver. The deck is not a standard card deck. Each "
            "round uses a 20-card deck: 6 Queens, 6 Kings, 6 Aces, and 2 "
            "Jokers. Each alive player is dealt 5 cards at the start of the "
            "round."
        )

    @staticmethod
    def _retry_note(error: str, context: AgentContext) -> str:
        return (
            "\n\nYour previous response was invalid and no action was taken.\n"
            f"Validation error: {error}\n"
            f"Legal cards: {context.legal_cards_text()}\n"
            f"Legal card counts: {context.legal_card_counts}\n"
            "Respond again using only the requested JSON schema."
        )

    @staticmethod
    def _reflection_retry_note(error: str, context: AgentContext) -> str:
        other_players = [name for name in context.alive_players if name != context.player_name]
        return (
            "\n\nYour previous reflection response was invalid and no memory was "
            "updated.\n"
            f"Validation error: {error}\n"
            "Return only valid JSON, with no markdown fences or commentary.\n"
            "Required shape: "
            '{"opinions": {"Opponent Name": "brief strategy impression"}}\n'
            f"Allowed opponent names: {', '.join(other_players) if other_players else 'none'}."
        )

    def _record(self, *, phase, context, requested, chosen, error, attempt):
        self.decision_log.append(
            {
                "player": context.player_name,
                "round_id": context.round_id,
                "phase": phase,
                "attempt": attempt,
                "requested": self._safe_payload(requested),
                "chosen": self._safe_payload(chosen),
                "error": error,
                "agent_type": type(self).__name__,
            }
        )

    @staticmethod
    def _safe_payload(value):
        if value is None:
            return None
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        if isinstance(value, (str, int, float, bool, list, dict)):
            return value
        return repr(value)

    def _sleep_after_llm_call(self):
        if self.llm_pause_seconds > 0:
            time.sleep(self.llm_pause_seconds)
