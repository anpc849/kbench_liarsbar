from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayDecision:
    """Agent decision for the card-play phase."""

    played_cards: list[str]
    behavior: str
    play_reason: str


@dataclass
class ChallengeDecision:
    """Agent decision for the challenge phase."""

    was_challenged: bool
    challenge_reason: str


@dataclass
class AgentContext:
    """Private, structured game context for one player at one decision point."""

    player_name: str
    phase: str
    round_id: int
    target_card: str | None
    hand: list[str]
    own_shots_taken: int
    revolver_chambers: int
    alive_players: list[str]
    public_history: list[dict[str, Any]]
    private_decision_history: list[dict[str, Any]] = field(default_factory=list)
    legal_cards: list[str] = field(default_factory=list)
    legal_card_counts: list[int] = field(default_factory=list)
    next_player: str | None = None
    previous_play: dict[str, Any] | None = None
    opinions: dict[str, str] = field(default_factory=dict)
    extra_hint: str = ""
    round_result: dict[str, Any] | None = None

    def legal_cards_text(self) -> str:
        return ", ".join(self.legal_cards) if self.legal_cards else "None"

    def public_history_text(self) -> str:
        if not self.public_history:
            return "- No public actions yet."
        rows = []
        for item in self.public_history[-12:]:
            kind = item.get("type", "event")
            if kind == "play":
                rows.append(
                    "- "
                    f"{item['player']} claimed {item['claimed_count']} "
                    f"{item['target_card']} card(s); "
                    f"remaining cards: {item['remaining_count']}; "
                    f"behavior: {item.get('behavior', '')}"
                )
            elif kind == "challenge":
                rows.append(
                    "- "
                    f"{item['challenger']} challenged {item['challenged_player']}; "
                    f"success={item['challenge_success']}; "
                    f"revealed={', '.join(item.get('revealed_cards', []))}"
                )
            elif kind == "no_challenge":
                rows.append(
                    "- "
                    f"{item['challenger']} did not challenge "
                    f"{item['challenged_player']}"
                )
            elif kind == "shot":
                rows.append(
                    "- "
                    f"{item['shooter']} fired; "
                    f"hit={item['bullet_hit']}; "
                    f"alive={item['shooter_alive']}"
                )
            elif kind == "round_start":
                rows.append(
                    "- "
                    f"Round {item['round_id']} started with target "
                    f"{item['target_card']}; players={', '.join(item['players'])}; "
                    f"starting_player={item['starting_player']}"
                )
            else:
                rows.append(f"- Public event: {kind}")
        return "\n".join(rows)

    def opinions_text(self) -> str:
        if not self.opinions:
            return "- No prior opinions."
        return "\n".join(
            f"- {name}: {opinion}" for name, opinion in sorted(self.opinions.items())
        )

    def private_decision_history_text(self) -> str:
        if not self.private_decision_history:
            return "- No prior private decisions."
        rows = []
        for item in self.private_decision_history[-8:]:
            phase = item.get("phase", "decision")
            decision = item.get("decision", {})
            if phase == "play":
                played = ", ".join(decision.get("played_cards", [])) or "None"
                rows.append(
                    "- "
                    f"Round {item.get('round_id')}, turn {item.get('turn')}: "
                    f"you played {played}. "
                    f"Private play_reason: {decision.get('play_reason', '')}"
                )
            elif phase == "challenge":
                action = (
                    "challenged"
                    if decision.get("was_challenged")
                    else "did not challenge"
                )
                rows.append(
                    "- "
                    f"Round {item.get('round_id')}, turn {item.get('turn')}: "
                    f"you {action}. "
                    "Private challenge_reason: "
                    f"{decision.get('challenge_reason', '')}"
                )
            else:
                rows.append(
                    "- "
                    f"Round {item.get('round_id')}, turn {item.get('turn')}: "
                    f"private {phase} decision."
                )
        return "\n".join(rows)

    def revolver_status_text(self) -> str:
        remaining = max(self.revolver_chambers - self.own_shots_taken, 1)
        risk_percent = 100 / remaining
        risk_text = f"{risk_percent:.1f}".rstrip("0").rstrip(".")
        return (
            f"{self.own_shots_taken} of {self.revolver_chambers} chambers have "
            "already been fired; each penalty shot advances to the next chamber. "
            "One hidden chamber contains the bullet, so your next penalty shot "
            f"has a {risk_text}% chance to eliminate you."
        )

    def to_text(self) -> str:
        previous = self.previous_play or {}
        previous_text = (
            "None"
            if not previous
            else (
                f"{previous.get('player')} claimed "
                f"{previous.get('claimed_count')} {previous.get('target_card')} "
                f"card(s), has {previous.get('remaining_count')} cards left, "
                f"behavior: {previous.get('behavior', '')}"
            )
        )
        return (
            f"Player: {self.player_name}\n"
            f"Phase: {self.phase}\n"
            f"Round: {self.round_id}\n"
            f"Target card: {self.target_card}\n"
            f"Your hand: {', '.join(self.hand) if self.hand else 'empty'}\n"
            f"Your revolver status: {self.revolver_status_text()}\n"
            f"Alive players: {', '.join(self.alive_players)}\n"
            f"Next player: {self.next_player or 'None'}\n"
            f"Legal cards to play: {self.legal_cards_text()}\n"
            f"Legal card counts: {self.legal_card_counts}\n"
            f"Previous play: {previous_text}\n"
            f"Extra hint: {self.extra_hint or 'None'}\n\n"
            f"Public history:\n{self.public_history_text()}\n\n"
            "Your private decision history:\n"
            f"{self.private_decision_history_text()}\n\n"
            f"Your private opinions:\n{self.opinions_text()}"
        )


class BaseAgent:
    """Base class for Liars Bar-compatible agents.

    User agents may subclass this class, but validation is behavioral: any
    object with the required methods can be used. The game must not inspect
    how an agent makes decisions internally.
    """

    def bind(self, participant, game):
        self.participant = participant
        self.game = game
        return self

    def setup(self) -> None:
        return None

    def choose_play(self, context: AgentContext) -> PlayDecision:
        raise NotImplementedError("Agents must implement choose_play().")

    def choose_challenge(self, context: AgentContext) -> ChallengeDecision:
        raise NotImplementedError("Agents must implement choose_challenge().")

    def reflect(self, context: AgentContext):
        return None

    def get_log(self) -> dict[str, Any] | None:
        return None

    def agent_name(self) -> str:
        return type(self).__name__

    def model_name(self) -> str:
        return ""
