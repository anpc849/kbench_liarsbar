from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from .agent.base import AgentContext
from .agent.llm_default import DefaultLLMAgent
from .agent.validation import (
    InvalidAgentError,
    bind_and_validate_agent,
    is_raw_kbench_llm,
    validate_challenge_decision,
    validate_play_decision,
)
from .config import GameConfig


CARDS = ("Q", "K", "A", "Joker")
TARGET_CARDS = ("Q", "K", "A")


@dataclass
class ParticipantState:
    name: str
    agent: Any
    seat: int
    model_id: str = ""
    evaluated: bool = False
    hand: list[str] = field(default_factory=list)
    alive: bool = True
    bullet_position: int = 0
    chamber_position: int = 0
    total_shots_taken: int = 0
    opinions: dict[str, str] = field(default_factory=dict)

    @property
    def shots_taken(self) -> int:
        return self.total_shots_taken


def player_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "player"


def coerce_game_config(game_config) -> GameConfig:
    if isinstance(game_config, GameConfig):
        config = game_config
    elif isinstance(game_config, dict):
        config = GameConfig(**game_config)
    else:
        raise TypeError(
            "game_config must be a GameConfig or dict; "
            f"got {type(game_config).__name__}"
        )
    if not 2 <= len(config.player_configs) <= 4:
        raise ValueError("Liars Bar requires 2 to 4 player_configs.")
    if config.revolver_chambers < 1:
        raise ValueError("GameConfig.revolver_chambers must be at least 1.")
    for index, spec in enumerate(config.player_configs, start=1):
        if "agent" not in spec:
            raise InvalidAgentError(
                f"player_configs[{index}] is missing required agent."
            )
    return config


class LiarsBarGame:
    """Source-faithful Liars Bar game runner with validated opaque agents."""

    def __init__(self, *, game_config: GameConfig | dict, UI=None):
        self.config = coerce_game_config(game_config)
        self.UI = UI
        self.rng = random.Random(self.config.seed)
        self.players: list[ParticipantState] = []
        self.agents: dict[str, Any] = {}
        self.current_player_idx = 0
        self.target_card: str | None = None
        self.round_id = 0
        self.turn_count = 0
        self.game_over = False
        self.timeout = False
        self.winner: str | None = None
        self.last_shooter_name: str | None = None
        self.public_history: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []
        self.replay_events: list[dict[str, Any]] = []
        self._setup_players()

    def _setup_players(self):
        names = set()
        for index, spec in enumerate(self.config.player_configs, start=1):
            name = str(spec.get("name") or f"Player {index}")
            if name in names:
                raise ValueError(f"Duplicate player name: {name}")
            names.add(name)
            participant = ParticipantState(
                name=name,
                agent=spec["agent"],
                seat=index - 1,
                model_id=str(spec.get("model_id", "")),
                evaluated=bool(spec.get("evaluated", False)),
                bullet_position=self.rng.randint(0, self.config.revolver_chambers - 1),
            )
            self.players.append(participant)
        for player in self.players:
            player.opinions = {
                other.name: "No prior observations."
                for other in self.players
                if other.name != player.name
            }
        for player in self.players:
            agent = self._normalize_agent(player.agent)
            self.agents[player.name] = bind_and_validate_agent(agent, player, self)

    @staticmethod
    def _normalize_agent(agent):
        if is_raw_kbench_llm(agent):
            return DefaultLLMAgent(agent)
        return agent

    def create_deck(self) -> list[str]:
        deck = ["Q"] * 6 + ["K"] * 6 + ["A"] * 6 + ["Joker"] * 2
        self.rng.shuffle(deck)
        return deck

    def deal_cards(self):
        deck = self.create_deck()
        for player in self.players:
            if player.alive:
                player.hand = []
        for _ in range(5):
            for player in self.players:
                if player.alive and deck:
                    player.hand.append(deck.pop())

    def start(self):
        self.current_player_idx = self.rng.randrange(len(self.players))
        self._reset_round(record_shooter=False, initial=True)
        self._emit_update("Game started.")
        while not self.game_over:
            self._check_stop()
            if self.round_id > self.config.max_rounds or self.turn_count >= self.config.max_turns:
                self.timeout = True
                self.game_over = True
                break
            self.play_turn()
        self._emit_update(self.winner or "Game ended.")
        return self.result_summary()

    def play_turn(self):
        self._check_stop()
        current = self.players[self.current_player_idx]
        if not current.alive:
            self.current_player_idx = self.find_next_alive_index(self.current_player_idx)
            return
        if self.check_other_players_no_cards(current):
            self.handle_system_challenge(current)
            return

        next_idx = self.find_next_player_with_cards(self.current_player_idx)
        next_player = self.players[next_idx]
        play_context = self.build_play_context(current, next_player)
        self._emit_update(f"{current.name} is choosing cards.")
        self._check_stop()
        current_agent = self.agents[current.name]
        play_raw = current_agent.choose_play(play_context)
        play = validate_play_decision(
            play_raw,
            current.hand,
            allow_empty_behavior=bool(getattr(current_agent, "allow_empty_behavior", False)),
        )
        for card in play.played_cards:
            current.hand.remove(card)
        self.turn_count += 1

        play_event = {
            "type": "play",
            "round_id": self.round_id,
            "turn": self.turn_count,
            "player": current.name,
            "target_card": self.target_card,
            "claimed_count": len(play.played_cards),
            "remaining_count": len(current.hand),
            "behavior": play.behavior,
        }
        self.public_history.append(play_event)
        self._record_replay(
            {
                **play_event,
                "private": False,
                "actual_cards_private": list(play.played_cards),
            }
        )
        self._record_decision(
            current,
            "play",
            play,
            private_payload={"actual_cards": list(play.played_cards)},
        )

        challenge_context = self.build_challenge_context(
            challenger=next_player,
            challenged=current,
            previous_play=play_event,
        )
        self._emit_update(f"{next_player.name} is deciding whether to challenge.")
        self._check_stop()
        challenge_agent = self.agents[next_player.name]
        challenge_raw = challenge_agent.choose_challenge(challenge_context)
        challenge = validate_challenge_decision(
            challenge_raw,
            allow_empty_reason=bool(getattr(challenge_agent, "allow_empty_challenge_reason", False)),
        )
        self._record_decision(next_player, "challenge", challenge)

        if challenge.was_challenged:
            valid = self.is_valid_play(play.played_cards)
            challenge_success = not valid
            challenge_event = {
                "type": "challenge",
                "round_id": self.round_id,
                "challenger": next_player.name,
                "challenged_player": current.name,
                "challenge_success": challenge_success,
                "revealed_cards": list(play.played_cards),
            }
            self.public_history.append(challenge_event)
            self._record_replay({**challenge_event, "private": False})
            loser = current if challenge_success else next_player
            self.perform_penalty(loser)
            return

        no_challenge_event = {
            "type": "no_challenge",
            "round_id": self.round_id,
            "challenger": next_player.name,
            "challenged_player": current.name,
        }
        self.public_history.append(no_challenge_event)
        self._record_replay({**no_challenge_event, "private": False})
        self.current_player_idx = next_idx

    def handle_system_challenge(self, current: ParticipantState):
        self._check_stop()
        actual_cards = list(current.hand)
        current.hand.clear()
        self.turn_count += 1
        play_event = {
            "type": "play",
            "round_id": self.round_id,
            "turn": self.turn_count,
            "player": current.name,
            "target_card": self.target_card,
            "claimed_count": len(actual_cards),
            "remaining_count": 0,
            "behavior": "System auto-played remaining cards because all other alive players had no cards.",
        }
        self.public_history.append(play_event)
        valid = self.is_valid_play(actual_cards)
        challenge_event = {
            "type": "challenge",
            "round_id": self.round_id,
            "challenger": "System",
            "challenged_player": current.name,
            "challenge_success": not valid,
            "revealed_cards": actual_cards,
            "reason": "System challenge: all other alive players had no cards.",
        }
        self.public_history.append(challenge_event)
        self._record_replay({**play_event, "private": False, "actual_cards_private": actual_cards})
        self._record_replay({**challenge_event, "private": False})
        if valid:
            shot_event = {
                "type": "shot",
                "round_id": self.round_id,
                "shooter": "None",
                "bullet_hit": False,
                "shooter_alive": True,
                "shots_taken": 0,
                "revolver_chambers": self.config.revolver_chambers,
            }
            self.public_history.append(shot_event)
            self._record_replay({**shot_event, "private": False})
            self._reset_round(record_shooter=False)
        else:
            self.perform_penalty(current)

    def perform_penalty(self, player: ParticipantState):
        self._check_stop()
        bullet_hit = player.bullet_position == player.chamber_position
        if bullet_hit:
            player.alive = False
        player.total_shots_taken += 1
        player.chamber_position = (
            player.chamber_position + 1
        ) % self.config.revolver_chambers
        self.last_shooter_name = player.name
        shot_event = {
            "type": "shot",
            "round_id": self.round_id,
            "shooter": player.name,
            "bullet_hit": bullet_hit,
            "shooter_alive": player.alive,
            "shots_taken": player.shots_taken,
            "revolver_chambers": self.config.revolver_chambers,
        }
        self.public_history.append(shot_event)
        self._record_replay({**shot_event, "private": False})
        if not self.check_victory():
            self._reset_round(record_shooter=True)

    def _reset_round(self, *, record_shooter: bool, initial: bool = False):
        if not initial and self.config.enable_reflection:
            self.handle_reflection()
        self.deal_cards()
        self.target_card = self.rng.choice(TARGET_CARDS)
        self.round_id += 1
        if not initial and record_shooter and self.last_shooter_name:
            shooter_idx = self.index_by_name(self.last_shooter_name)
            if shooter_idx is not None and self.players[shooter_idx].alive:
                self.current_player_idx = shooter_idx
            else:
                self.current_player_idx = self.find_next_alive_index(shooter_idx or 0)
        elif not initial:
            alive = [p for p in self.players if p.alive]
            self.current_player_idx = self.players.index(self.rng.choice(alive))

        event = {
            "type": "round_start",
            "round_id": self.round_id,
            "target_card": self.target_card,
            "players": [p.name for p in self.players if p.alive],
            "starting_player": self.players[self.current_player_idx].name,
        }
        self.public_history.append(event)
        self._record_replay(
            {
                **event,
                "private": False,
                "private_initial_state": [
                    {
                        "player": p.name,
                        "hand": list(p.hand),
                        "bullet_position": p.bullet_position,
                        "chamber_position": p.chamber_position,
                        "revolver_chambers": self.config.revolver_chambers,
                    }
                    for p in self.players
                    if p.alive
                ],
            }
        )

    def handle_reflection(self):
        for player in [p for p in self.players if p.alive]:
            self._emit_update(f"{player.name} is updating reflection memory.")
            self._check_stop()
            context = self.build_reflection_context(player)
            updated = self.agents[player.name].reflect(context)
            if isinstance(updated, dict):
                player.opinions.update(
                    {
                        str(name): str(opinion)
                        for name, opinion in updated.items()
                        if name != player.name
                    }
                )

    def is_valid_play(self, cards: list[str]) -> bool:
        return all(card == self.target_card or card == "Joker" for card in cards)

    def check_victory(self) -> bool:
        alive = [p for p in self.players if p.alive]
        if len(alive) == 1:
            self.winner = alive[0].name
            self.game_over = True
            return True
        return False

    def check_other_players_no_cards(self, current: ParticipantState) -> bool:
        others = [p for p in self.players if p is not current and p.alive]
        return bool(others) and all(not p.hand for p in others)

    def find_next_player_with_cards(self, start_idx: int) -> int:
        idx = start_idx
        for _ in range(len(self.players)):
            idx = (idx + 1) % len(self.players)
            player = self.players[idx]
            if player.alive and player.hand:
                return idx
        return start_idx

    def find_next_alive_index(self, start_idx: int) -> int:
        idx = start_idx
        for _ in range(len(self.players)):
            idx = (idx + 1) % len(self.players)
            if self.players[idx].alive:
                return idx
        return start_idx

    def index_by_name(self, name: str) -> int | None:
        for index, player in enumerate(self.players):
            if player.name == name:
                return index
        return None

    def build_play_context(self, player: ParticipantState, next_player: ParticipantState):
        return AgentContext(
            player_name=player.name,
            phase="play",
            round_id=self.round_id,
            target_card=self.target_card,
            hand=list(player.hand),
            own_shots_taken=player.shots_taken,
            revolver_chambers=self.config.revolver_chambers,
            alive_players=[p.name for p in self.players if p.alive],
            public_history=list(self.public_history),
            private_decision_history=self._private_decision_history_for(player.name),
            legal_cards=list(player.hand),
            legal_card_counts=list(range(1, min(3, len(player.hand)) + 1)),
            next_player=next_player.name,
            opinions=dict(player.opinions),
        )

    def build_challenge_context(
        self,
        *,
        challenger: ParticipantState,
        challenged: ParticipantState,
        previous_play: dict[str, Any],
    ):
        extra_hint = (
            "All other alive players have no cards."
            if self.check_other_players_no_cards(challenger)
            else ""
        )
        return AgentContext(
            player_name=challenger.name,
            phase="challenge",
            round_id=self.round_id,
            target_card=self.target_card,
            hand=list(challenger.hand),
            own_shots_taken=challenger.shots_taken,
            revolver_chambers=self.config.revolver_chambers,
            alive_players=[p.name for p in self.players if p.alive],
            public_history=list(self.public_history),
            private_decision_history=self._private_decision_history_for(challenger.name),
            previous_play=dict(previous_play),
            next_player=challenged.name,
            opinions=dict(challenger.opinions),
            extra_hint=extra_hint,
        )

    def build_reflection_context(self, player: ParticipantState):
        return AgentContext(
            player_name=player.name,
            phase="reflect",
            round_id=self.round_id,
            target_card=self.target_card,
            hand=list(player.hand),
            own_shots_taken=player.shots_taken,
            revolver_chambers=self.config.revolver_chambers,
            alive_players=[p.name for p in self.players if p.alive],
            public_history=list(self.public_history),
            private_decision_history=self._private_decision_history_for(player.name),
            opinions=dict(player.opinions),
        )

    def _private_decision_history_for(self, player_name: str) -> list[dict[str, Any]]:
        return [
            dict(event)
            for event in self.decision_log
            if player_name in event.get("visible_to", [])
        ]

    def result_summary(self) -> dict[str, Any]:
        has_human = any(
            (self._model_name(self.agents[p.name]) or p.model_id) == "human"
            for p in self.players
        )
        players = []
        for p in self.players:
            agent = self.agents[p.name]
            model = self._model_name(agent) or p.model_id
            player_payload = {
                "id": player_id(p.name),
                "name": p.name,
                "alive": p.alive,
                "seat": p.seat,
                "shots_taken": p.shots_taken,
                "hand_count": len(p.hand),
                "agent_type": self._agent_type_name(agent),
                "model": model,
                "evaluated": p.evaluated,
            }
            if not has_human or model == "human":
                player_payload["hand"] = list(p.hand)
            players.append(player_payload)

        return {
            "winner": self.winner,
            "timeout": self.timeout,
            "rounds": self.round_id,
            "turns": self.turn_count,
            "revolver_chambers": self.config.revolver_chambers,
            "players": players,
            "public_history": list(self.public_history),
            "decision_log": list(self.decision_log),
            "game_log": {
                "schema_version": "liarsbar-game-log-v1",
                "seed": self.config.seed,
                "winner": self.winner,
                "timeout": self.timeout,
                "events": list(self.replay_events),
            },
        }

    def _record_decision(self, player: ParticipantState, phase: str, decision, private_payload=None):
        payload = {
            "player": player.name,
            "round_id": self.round_id,
            "turn": self.turn_count,
            "phase": phase,
            "decision": dict(decision.__dict__) if hasattr(decision, "__dict__") else repr(decision),
            "private": True,
            "visible_to": [player.name],
        }
        if private_payload:
            payload.update(private_payload)
        self.decision_log.append(payload)
        self._record_replay({**payload, "type": "agent_decision"})

    def _record_replay(self, event: dict[str, Any]):
        event = dict(event)
        event.setdefault("event_id", len(self.replay_events) + 1)
        self.replay_events.append(event)
        self._emit_update("")

    def _check_stop(self):
        if self.UI is not None and callable(getattr(self.UI, "check_stop", None)):
            self.UI.check_stop()

    def _emit_update(self, report: str):
        if self.UI is None:
            return
        if report and callable(getattr(self.UI, "report", None)):
            self.UI.report(report)
        if callable(getattr(self.UI, "draw_game", None)):
            self.UI.draw_game(self)

    @staticmethod
    def _agent_type_name(agent) -> str:
        if callable(getattr(agent, "agent_name", None)):
            return str(agent.agent_name())
        return type(agent).__name__

    @staticmethod
    def _model_name(agent) -> str:
        if callable(getattr(agent, "model_name", None)):
            return str(agent.model_name())
        return ""


def run_liarsbar_game(*, game_config: GameConfig | dict, UI=None):
    return LiarsBarGame(game_config=game_config, UI=UI).start()
