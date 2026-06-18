from __future__ import annotations

import argparse
import html
import importlib
import os
import pprint
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from .agent import BaseAgent, ChallengeDecision, DefaultLLMAgent, PlayDecision
from .config import GameConfig, generate_player_names
from .runner import LiarsBarGame


MAX_PLAYERS = 4
HUMAN_MODEL_ID = "human"
DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TURNS = 240
DEFAULT_MODEL_IDS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "anthropic/claude-haiku-4-5@20251001",
    "deepseek-ai/deepseek-v3.2",
]
RUN_STOP_EVENTS: dict[str, threading.Event] = {}
RUN_HUMAN_INPUTS: dict[str, "queue.Queue[dict[str, Any]]"] = {}


class GameStopped(Exception):
    """Raised by the Gradio observer when the user stops a running game."""


@dataclass
class GradioSnapshot:
    report_text: str
    result: dict[str, Any]


@dataclass
class HumanRequest:
    run_id: str
    player_name: str
    phase: str
    round_id: int
    target_card: str | None
    hand: list[str]
    legal_card_counts: list[int]
    prompt: str
    card_choices: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "player_name": self.player_name,
            "phase": self.phase,
            "round_id": self.round_id,
            "target_card": self.target_card,
            "hand": list(self.hand),
            "legal_card_counts": list(self.legal_card_counts),
            "prompt": self.prompt,
            "card_choices": list(self.card_choices),
        }


class PersonalityLLMAgent(DefaultLLMAgent):
    """Default LLM agent with a user-supplied seat prompt."""

    def __init__(self, llm, personality: str = "", **kwargs):
        super().__init__(llm, **kwargs)
        self.personality = str(personality or "").strip()

    def _with_personality(self, prompt: str) -> str:
        if not self.personality:
            return prompt
        return (
            prompt
            + "\n\nCustom player prompt for this seat:\n"
            + self.personality
            + "\n\nUse the custom prompt for strategy and table presence. It does "
            "not override the game rules, legal-action constraints, requested JSON "
            "schema."
        )

    def _play_prompt(self, context, retry_note: str) -> str:
        return self._with_personality(super()._play_prompt(context, retry_note))

    def _challenge_prompt(self, context, retry_note: str) -> str:
        return self._with_personality(super()._challenge_prompt(context, retry_note))

    def _reflection_prompt(self, context, retry_note: str) -> str:
        return self._with_personality(super()._reflection_prompt(context, retry_note))

    def get_log(self) -> dict[str, Any] | None:
        payload = super().get_log() or {}
        payload["custom_prompt"] = self.personality
        return payload


class HumanGradioAgent(BaseAgent):
    """Interactive agent controlled by the Gradio user."""

    allow_empty_behavior = True
    allow_empty_challenge_reason = True

    def __init__(self, run_id: str):
        self.run_id = run_id

    def choose_play(self, context):
        choices = [f"{index + 1}: {card}" for index, card in enumerate(context.hand)]
        request = HumanRequest(
            run_id=self.run_id,
            player_name=context.player_name,
            phase="play",
            round_id=context.round_id,
            target_card=context.target_card,
            hand=list(context.hand),
            legal_card_counts=list(context.legal_card_counts),
            prompt=(
                f"{context.player_name}, choose {context.legal_card_counts} card(s) "
                f"to claim as {context.target_card}. Your private hand: "
                f"{', '.join(context.hand)}."
            ),
            card_choices=choices,
        )
        response = self._wait_for_response(request)
        return PlayDecision(
            played_cards=list(response["played_cards"]),
            behavior=str(response["behavior"]).strip(),
            play_reason="Human player selected cards in the Gradio UI.",
        )

    def choose_challenge(self, context):
        previous = context.previous_play or {}
        request = HumanRequest(
            run_id=self.run_id,
            player_name=context.player_name,
            phase="challenge",
            round_id=context.round_id,
            target_card=context.target_card,
            hand=list(context.hand),
            legal_card_counts=[],
            prompt=(
                f"{context.player_name}, decide whether to challenge "
                f"{previous.get('player')} claiming {previous.get('claimed_count')} "
                f"{previous.get('target_card')} card(s). Your private hand: "
                f"{', '.join(context.hand)}."
            ),
            card_choices=[],
        )
        response = self._wait_for_response(request)
        return ChallengeDecision(
            was_challenged=bool(response["was_challenged"]),
            challenge_reason=str(response["challenge_reason"]).strip(),
        )

    def reflect(self, context):
        return None

    def agent_name(self) -> str:
        return type(self).__name__

    def model_name(self) -> str:
        return HUMAN_MODEL_ID

    def get_log(self) -> dict[str, Any] | None:
        return {"human": True}

    def _wait_for_response(self, request: HumanRequest) -> dict[str, Any]:
        ui = getattr(getattr(self, "game", None), "UI", None)
        if ui is None or not callable(getattr(ui, "request_human_decision", None)):
            raise RuntimeError("Human players require the Gradio app.")
        ui.request_human_decision(request)
        inputs = RUN_HUMAN_INPUTS.setdefault(self.run_id, queue.Queue())
        while True:
            if callable(getattr(ui, "check_stop", None)):
                ui.check_stop()
            try:
                response = inputs.get(timeout=0.2)
            except queue.Empty:
                continue
            if response.get("phase") == request.phase and response.get("player_name") == request.player_name:
                return response


class GradioGameUI:
    """Observer used by the Liars Bar runner to stream snapshots to Gradio."""

    def __init__(self, updates: "queue.Queue[GradioSnapshot]", stop_event: threading.Event):
        self.updates = updates
        self.stop_event = stop_event
        self.report_text = ""

    def check_stop(self):
        if self.stop_event.is_set():
            raise GameStopped("Game stopped by user.")

    def report(self, text: str):
        self.check_stop()
        if text:
            self.report_text = text

    def draw_game(self, game: LiarsBarGame):
        self.check_stop()
        self.updates.put(
            GradioSnapshot(
                report_text=self.report_text,
                result=game.result_summary(),
            )
        )

    def request_human_decision(self, request: HumanRequest):
        self.check_stop()
        self.updates.put(request)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_kbench():
    local_src = workspace_root() / "kaggle-benchmarks" / "src"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))

    import kaggle_benchmarks as kbench

    try:
        if len(list(getattr(kbench, "llms", {}).keys())) > 0:
            return kbench
    except Exception as exc:
        raise RuntimeError("Unable to inspect kbench.llms.") from exc

    load_env_file(workspace_root() / "kaggle-benchmarks" / ".env")
    kbench = importlib.reload(kbench)
    try:
        if len(list(getattr(kbench, "llms", {}).keys())) > 0:
            return kbench
    except Exception as exc:
        raise RuntimeError("Unable to inspect kbench.llms after loading .env.") from exc
    raise RuntimeError("kbench.llms is empty after loading local .env.")


def model_choices(kbench) -> list[str]:
    choices = list(getattr(kbench, "llms", {}).keys())
    if not choices:
        choices = list(DEFAULT_MODEL_IDS)
    if HUMAN_MODEL_ID not in choices:
        choices.append(HUMAN_MODEL_ID)
    return choices


def default_state(choices: list[str]) -> dict[str, Any]:
    names = generate_player_names(MAX_PLAYERS)
    return {
        "player_count": 4,
        "evaluated_index": 0,
        "revolver_chambers": 6,
        "enable_reflection": True,
        "players": [
            {
                "name": names[index],
                "model": choices[min(index, len(choices) - 1)] if choices else "",
                "personality": "",
            }
            for index in range(MAX_PLAYERS)
        ],
    }


def collect_state(*values) -> dict[str, Any]:
    player_count = int(values[0])
    evaluated_index = 0
    revolver_chambers = int(values[1])
    enable_reflection = bool(values[2])
    players = []
    cursor = 3
    for index in range(MAX_PLAYERS):
        players.append(
            {
                "name": str(values[cursor]).strip(),
                "model": str(values[cursor + 1]).strip(),
                "personality": str(values[cursor + 2] or "").strip(),
            }
        )
        cursor += 3
    return {
        "player_count": player_count,
        "evaluated_index": evaluated_index,
        "revolver_chambers": revolver_chambers,
        "enable_reflection": enable_reflection,
        "players": players[:player_count],
    }


def validate_state(state: dict[str, Any]) -> None:
    if not 2 <= state["player_count"] <= 4:
        raise ValueError("Player count must be between 2 and 4.")
    if not 0 <= state["evaluated_index"] < state["player_count"]:
        raise ValueError("Evaluated seat must be within the active player count.")
    if state["revolver_chambers"] < 1:
        raise ValueError("Revolver chambers must be at least 1.")
    names = [player["name"] for player in state["players"]]
    if any(not name for name in names):
        raise ValueError("Every active player needs a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("Active player names must be distinct.")
    if any(not player["model"] for player in state["players"]):
        raise ValueError("Every active player needs a model.")


def make_game_config(kbench, state: dict[str, Any]) -> GameConfig:
    validate_state(state)
    player_configs = []
    for index, player in enumerate(state["players"]):
        if player["model"] == HUMAN_MODEL_ID:
            agent = HumanGradioAgent(str(state.get("run_id") or ""))
        else:
            llm = kbench.llms[player["model"]]
            agent = PersonalityLLMAgent(
                llm,
                personality=player.get("personality", ""),
            )
        player_configs.append(
            {
                "name": player["name"],
                "agent": agent,
                "model_id": player["model"],
                "evaluated": index == state["evaluated_index"],
            }
        )
    return GameConfig(
        player_configs=player_configs,
        seed=None,
        max_rounds=DEFAULT_MAX_ROUNDS,
        max_turns=DEFAULT_MAX_TURNS,
        revolver_chambers=state["revolver_chambers"],
        enable_reflection=state["enable_reflection"],
        evaluated_player_name=state["players"][state["evaluated_index"]]["name"],
    )


def export_config_payload(state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    return {
        "settings": {
            "seed": None,
            "max_rounds": DEFAULT_MAX_ROUNDS,
            "max_turns": DEFAULT_MAX_TURNS,
            "player_count": state["player_count"],
            "evaluated_index": state["evaluated_index"],
            "revolver_chambers": state["revolver_chambers"],
            "enable_reflection": state["enable_reflection"],
        },
        "players": list(state["players"]),
    }


def game_config_from_export(payload: dict[str, Any], kbench) -> GameConfig:
    player_configs = []
    evaluated_index = int(payload["settings"]["evaluated_index"])
    seed_value = payload["settings"].get("seed")
    for index, player in enumerate(payload["players"]):
        model_id = player.get("model_id") or player.get("model")
        if model_id == HUMAN_MODEL_ID:
            raise RuntimeError("Human players require the Gradio app and cannot be rebuilt from export.")
        player_configs.append(
            {
                "name": player["name"],
                "agent": PersonalityLLMAgent(
                    kbench.llms[model_id],
                    personality=player.get("personality", ""),
                ),
                "model_id": model_id,
                "evaluated": index == evaluated_index,
            }
        )
    return GameConfig(
        player_configs=player_configs,
        seed=None if seed_value is None else int(seed_value),
        max_rounds=int(payload["settings"].get("max_rounds", DEFAULT_MAX_ROUNDS)),
        max_turns=int(payload["settings"].get("max_turns", DEFAULT_MAX_TURNS)),
        revolver_chambers=int(payload["settings"]["revolver_chambers"]),
        enable_reflection=bool(payload["settings"]["enable_reflection"]),
        evaluated_player_name=payload["players"][evaluated_index]["name"],
    )


def export_config_code(state: dict[str, Any]) -> str:
    payload = export_config_payload(state)
    return (
        "import kbench_liarsbar as liarsbar\n"
        "from kbench_liarsbar.gradio_app import game_config_from_export\n\n"
        "game_config_payload = "
        + pprint.pformat(payload, width=100, sort_dicts=False)
        + "\n\n"
        "game_config = game_config_from_export(game_config_payload, kbench)\n"
        "result = liarsbar.run_liarsbar_game(game_config=game_config)\n"
    )


def randomize_names(player_count: int):
    names = generate_player_names(MAX_PLAYERS, seed=time.time_ns() % 1_000_000)
    return [gr.update(value=names[index]) for index in range(MAX_PLAYERS)]


def update_player_visibility(count):
    count = int(count)
    return [gr.update(visible=index < count) for index in range(MAX_PLAYERS)]


def empty_display(message: str = "Configure and run a game."):
    return (
        "<section class='lb-board empty'>"
        f"<div class='lb-empty'>{html.escape(message)}</div>"
        "</section>"
    )


def short_model_name(model: str) -> str:
    value = str(model or "")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if "@" in value:
        value = value.split("@", 1)[0]
    return value[:26]


def latest_public_event(public_history: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in reversed(public_history):
        if event.get("type") == event_type:
            return event
    return None


def latest_round_label(public_history: list[dict[str, Any]]) -> str:
    event = latest_public_event(public_history, "round_start")
    if not event:
        return "Waiting for first round"
    return f"Round {event.get('round_id')} - target {event.get('target_card')}"


def player_color(name: str) -> str:
    colors = [
        "#60a5fa",
        "#f59e0b",
        "#34d399",
        "#f472b6",
        "#a78bfa",
        "#fb7185",
    ]
    value = sum(ord(char) for char in str(name))
    return colors[value % len(colors)]


def card_html(rank: str = "", *, hidden: bool = False, small: bool = False) -> str:
    classes = ["lb-card"]
    if hidden:
        classes.append("back")
    if small:
        classes.append("small")
    label = "?" if hidden else str(rank or "?")
    rank_key = label.lower()
    if label == "Joker":
        label = "J"
        rank_key = "joker"
    if not hidden:
        classes.append(f"rank-{rank_key}")
    return (
        f"<span class='{' '.join(classes)}'>"
        f"<span class='rank'>{html.escape(label)}</span>"
        "</span>"
    )


def card_stack_html(count: int) -> str:
    safe_count = max(0, min(int(count or 0), 5))
    return "<div class='lb-hand-stack'>" + "".join(
        card_html(hidden=True, small=True) for _ in range(safe_count)
    ) + "</div>"


def hand_stack_html(player: dict[str, Any]) -> str:
    hand = player.get("hand")
    if isinstance(hand, list):
        safe_hand = [str(card) for card in hand[:5]]
        return "<div class='lb-hand-stack face-up'>" + "".join(
            card_html(card, small=True) for card in safe_hand
        ) + "</div>"
    return card_stack_html(player.get("hand_count") or 0)


def claimed_cards_html(event: dict[str, Any] | None) -> str:
    if not event:
        return "<div class='lb-pot-cards muted'>No claim yet</div>"
    count = max(0, min(int(event.get("claimed_count") or 0), 3))
    return "<div class='lb-pot-cards'>" + "".join(
        card_html(hidden=True) for _ in range(count)
    ) + "</div>"


def revealed_cards_html(event: dict[str, Any] | None) -> str:
    if not event or not event.get("revealed_cards"):
        return ""
    cards = "".join(card_html(str(card), small=True) for card in event.get("revealed_cards", []))
    return f"<div class='lb-revealed'><span>Actual cards</span>{cards}</div>"


def current_round_events(public_history: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    round_start = latest_public_event(public_history, "round_start")
    if not round_start:
        return []
    round_id = round_start.get("round_id")
    return [
        (index, event)
        for index, event in enumerate(public_history)
        if event.get("round_id") == round_id
    ]


def latest_claim_state(
    public_history: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    events = current_round_events(public_history)
    latest_play_pair = next(
        ((index, event) for index, event in reversed(events) if event.get("type") == "play"),
        None,
    )
    if latest_play_pair is None:
        return None, None
    latest_play_index, latest_play = latest_play_pair
    resolution = next(
        (
            event
            for index, event in events
            if index > latest_play_index
            and event.get("type") in {"challenge", "no_challenge"}
        ),
        None,
    )
    return latest_play, resolution


def latest_shot_state(public_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not public_history:
        return None
    event = public_history[-1]
    if event.get("type") != "shot":
        return None
    return event


def shot_overlay_html(event: dict[str, Any] | None, revolver_chambers: int | None) -> str:
    if not event or event.get("shooter") == "None":
        return ""
    hit = bool(event.get("bullet_hit"))
    label = "BANG" if hit else "CLICK"
    shots_taken = int(event.get("shots_taken") or 0)
    chambers = int(event.get("revolver_chambers") or revolver_chambers or 0)
    progress = f"{shots_taken}/{chambers}" if chambers else str(shots_taken)
    subtext = (
        f"{event.get('shooter')} is dead. Shot {progress}."
        if hit
        else f"{event.get('shooter')} survived. Shot {progress}."
    )
    state_class = "hit" if hit else "safe"
    return (
        f"<div class='lb-shot-overlay {state_class}'>"
        "<div class='lb-revolver'>"
        "<span></span><span></span><span></span><span></span><span></span><span></span>"
        "</div>"
        f"<div class='lb-shot-word'>{html.escape(label)}</div>"
        f"<div class='lb-shot-progress'>{html.escape(progress)}</div>"
        f"<div class='lb-shot-subtext'>{html.escape(subtext)}</div>"
        "</div>"
    )


def render_game(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None:
        return empty_display()
    result = snapshot.result
    public_history = result.get("public_history", [])
    revolver_chambers = int(result.get("revolver_chambers") or 0)
    latest_round = latest_round_label(public_history)
    latest_play, latest_resolution = latest_claim_state(public_history)
    latest_shot = latest_shot_state(public_history)
    players = result.get("players", [])
    seats = []
    for index, player in enumerate(players):
        state_class = "alive" if player.get("alive") else "out"
        evaluated = " evaluated" if player.get("evaluated") else ""
        active = ""
        if latest_play and latest_play.get("player") == player.get("name"):
            active = " active"
        shot_class = ""
        if latest_shot and latest_shot.get("shooter") == player.get("name"):
            shot_class = " shot-hit" if latest_shot.get("bullet_hit") else " shot-safe"
        seat_class = f"seat-{len(players)}-{index + 1}"
        status = "Alive" if player.get("alive") else "Dead"
        shots_taken = int(player.get("shots_taken") or 0)
        shot_progress = f"{shots_taken}/{revolver_chambers}" if revolver_chambers else str(shots_taken)
        dead_stamp = "<div class='lb-dead-stamp'>DEAD</div>" if not player.get("alive") else ""
        status_code = "IN" if player.get("alive") else "OUT"
        seats.append(
            f"<article class='lb-player-seat {seat_class} {state_class}{evaluated}{active}{shot_class}'>"
            f"{hand_stack_html(player)}"
            f"{dead_stamp}"
            f"<div class='lb-name'>{html.escape(player['name'])}</div>"
            f"<div class='lb-model'>{html.escape(short_model_name(player.get('model') or ''))}</div>"
            "<div class='lb-seat-stats'>"
            f"<span class='lb-stat lb-stat-life' title='{html.escape(status)}'>{html.escape(status_code)}</span>"
            f"<span class='lb-stat lb-stat-shot' title='Shots'>{html.escape(shot_progress)}</span>"
            f"<span class='lb-stat lb-stat-card' title='Cards'>{player.get('hand_count')}</span>"
            "</div>"
            "</article>"
        )
    winner = result.get("winner")
    banner = snapshot.report_text or (f"Winner: {winner}" if winner else "Game running")
    target_card = ""
    round_start = latest_public_event(public_history, "round_start")
    if round_start:
        target_card = str(round_start.get("target_card") or "")
    claim_text = "No public claim yet."
    if latest_play:
        claim_text = (
            f"{latest_play.get('player')} claimed {latest_play.get('claimed_count')} "
            f"{latest_play.get('target_card')} card(s)."
        )
    challenge_text = ""
    if latest_resolution and latest_resolution.get("type") == "challenge":
        challenge_text = (
            f"{latest_resolution.get('challenger')} challenged "
            f"{latest_resolution.get('challenged_player')} - "
            f"{'success' if latest_resolution.get('challenge_success') else 'failed'}."
        )
    elif latest_resolution and latest_resolution.get("type") == "no_challenge":
        challenge_text = (
            f"{latest_resolution.get('challenger')} did not challenge "
            f"{latest_resolution.get('challenged_player')}."
        )
    return (
        "<section class='lb-board'>"
        f"<div class='lb-table-wrap'>"
        "<div class='lb-table-felt'>"
        f"{''.join(seats)}"
        "<div class='lb-pot'>"
        f"<div class='lb-round'>{html.escape(latest_round)}</div>"
        f"<div class='lb-target'>{card_html(target_card) if target_card else ''}<span>Target</span></div>"
        f"{claimed_cards_html(latest_play)}"
        f"<p>{html.escape(claim_text)}</p>"
        f"<p class='challenge'>{html.escape(challenge_text)}</p>"
        f"{revealed_cards_html(latest_resolution if latest_resolution and latest_resolution.get('type') == 'challenge' else None)}"
        f"{shot_overlay_html(latest_shot, revolver_chambers)}"
        "</div>"
        "</div>"
        "</div>"
        "<footer class='lb-table-footer'>"
        f"<div class='lb-footer-main'><h2>{html.escape(banner)}</h2><p>{html.escape(latest_round)}</p></div>"
        f"<div class='lb-counts'><span title='Rounds'>R {result.get('rounds')}</span><span title='Turns'>T {result.get('turns')}</span>"
        f"<span title='Revolver chambers'>C {result.get('revolver_chambers')}</span></div>"
        "</footer>"
        "</section>"
    )


def render_status(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None:
        return "<div class='lb-side'><h3>Status</h3><p>No game has run yet.</p></div>"
    result = snapshot.result
    winner = result.get("winner") or "Pending"
    evaluated = next((p for p in result.get("players", []) if p.get("evaluated")), None)
    eval_status = "won" if evaluated and winner == evaluated["name"] else "not won"
    return (
        "<div class='lb-side'>"
        "<h3>Status</h3>"
        f"<p><b>Winner:</b> {html.escape(str(winner))}</p>"
        f"<p><b>Evaluated:</b> {html.escape(evaluated['name'] if evaluated else 'None')} ({eval_status})</p>"
        f"<p><b>Report:</b> {html.escape(snapshot.report_text or '')}</p>"
        "</div>"
    )


def result_has_human(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    return any(str(player.get("model", "")).lower() == HUMAN_MODEL_ID for player in result.get("players", []))


def state_has_human(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    return any(str(player.get("model", "")).lower() == HUMAN_MODEL_ID for player in state.get("players", []))


def public_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        "winner": result.get("winner"),
        "timeout": result.get("timeout"),
        "rounds": result.get("rounds"),
        "turns": result.get("turns"),
        "revolver_chambers": result.get("revolver_chambers"),
        "players": list(result.get("players", [])),
        "public_history": list(result.get("public_history", [])),
    }


def result_json_update(result: dict[str, Any] | None, has_human: bool):
    payload = public_result(result) if has_human else (result or {})
    return gr.update(value=payload, visible=not has_human)


def event_actor(event: dict[str, Any]) -> str:
    return str(
        event.get("player")
        or event.get("challenger")
        or event.get("shooter")
        or event.get("starting_player")
        or "System"
    )


def conversation_message(event: dict[str, Any], *, include_reason: bool = True) -> tuple[str, str, str]:
    kind = str(event.get("type", "event"))
    actor = event_actor(event)
    meta = f"Round {event.get('round_id', '')}".strip()
    if kind == "round_start":
        return (
            "System",
            f"Round {event.get('round_id')} begins. Target card is {event.get('target_card')}. "
            f"{event.get('starting_player')} starts.",
            "Round start",
        )
    if kind == "play":
        behavior = str(event.get("behavior") or "").strip()
        claim = (
            f"I claim {event.get('claimed_count')} {event.get('target_card')} "
            "card(s)."
        )
        message = f"{claim} {behavior}".strip()
        return actor, message, meta
    if kind == "challenge":
        outcome = "succeeded" if event.get("challenge_success") else "failed"
        revealed = ", ".join(str(card) for card in event.get("revealed_cards", []))
        message = (
            f"I challenge {event.get('challenged_player')}. Challenge {outcome}. "
            f"Actual cards: {revealed or 'none'}."
        )
        reason = str(event.get("reason") or "").strip() if include_reason else ""
        if reason:
            message += f" {reason}"
        return actor, message, meta
    if kind == "no_challenge":
        message = f"I do not challenge {event.get('challenged_player')}."
        reason = str(event.get("reason") or "").strip() if include_reason else ""
        if reason:
            message += f" {reason}"
        return actor, message, meta
    if kind == "shot":
        if event.get("shooter") == "None":
            return "System", "No player fired this round.", meta
        hit = "hit" if event.get("bullet_hit") else "safe"
        alive = "still alive" if event.get("shooter_alive") else "dead"
        shots_taken = int(event.get("shots_taken") or 0)
        chambers = int(event.get("revolver_chambers") or 0)
        progress = f"{shots_taken}/{chambers}" if chambers else str(shots_taken)
        return (
            "System",
            f"{event.get('shooter')} fired the revolver: {hit}, {alive}. Shot {progress}.",
            meta,
        )
    return actor, str(event), meta


def render_conversation(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None:
        return (
            "<div class='lb-conversation'>"
            "<div class='lb-conversation-title'>Public action log</div>"
            "<div class='lb-empty-chat'>No public actions yet.</div>"
            "</div>"
        )
    public_history = snapshot.result.get("public_history", [])[-80:]
    include_reason = not result_has_human(snapshot.result)
    bubbles = []
    for event in public_history:
        actor, message, meta = conversation_message(event, include_reason=include_reason)
        is_system = actor == "System"
        color = "#94a3b8" if is_system else player_color(actor)
        system_class = " system" if is_system else ""
        bubbles.append(
            f"<div class='lb-chat-bubble{system_class}'>"
            f"<div class='lb-chat-dot' style='background:{html.escape(color)}'></div>"
            "<div class='lb-chat-body'>"
            "<div class='lb-chat-head'>"
            f"<span>{html.escape(actor)}</span>"
            f"<small>{html.escape(meta)}</small>"
            "</div>"
            f"<div class='lb-chat-message'>{html.escape(message)}</div>"
            "</div>"
            "</div>"
        )
    if not bubbles:
        bubbles.append("<div class='lb-empty-chat'>No public actions yet.</div>")
    return (
        "<div class='lb-conversation'>"
        "<div class='lb-conversation-title'>Public action log</div>"
        + "".join(bubbles)
        + "</div>"
    )


def render_side(snapshot: GradioSnapshot | None) -> str:
    return render_conversation(snapshot) + render_status(snapshot)


def render_side_notice(snapshot: GradioSnapshot | None, title: str, message: str) -> str:
    return (
        render_conversation(snapshot)
        + "<div class='lb-side'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(message)}</p>"
        "</div>"
    )


def hidden_human_controls():
    return [
        gr.update(visible=False),
        "",
        gr.update(choices=[], value=[], visible=False),
        gr.update(value="", label="Public table message", interactive=False),
        gr.update(choices=[], value=None, visible=False, interactive=False, label=""),
        gr.update(interactive=False),
        "",
    ]


def visible_human_controls(request: HumanRequest):
    if request.phase == "play":
        action_label = "Public table behavior"
        cards_update = gr.update(
            choices=request.card_choices,
            value=[],
            visible=True,
            label=f"Choose cards to play as {request.target_card}",
        )
        challenge_update = gr.update(
            choices=[],
            value=None,
            visible=False,
            interactive=False,
            label="",
        )
    else:
        action_label = "Public challenge comment"
        cards_update = gr.update(choices=[], value=[], visible=False)
        challenge_update = gr.update(
            choices=[
                ("Do not challenge", "no_challenge"),
                ("Challenge", "challenge"),
            ],
            value="no_challenge",
            visible=True,
            interactive=True,
            label="Challenge decision",
        )
    prompt = (
        "<div class='lb-human-prompt'>"
        f"<h3>{html.escape(request.player_name)}: {html.escape(request.phase.title())}</h3>"
        f"<p>{html.escape(request.prompt)}</p>"
        "</div>"
    )
    return [
        gr.update(visible=True),
        prompt,
        cards_update,
        gr.update(value="", label=action_label, interactive=True),
        challenge_update,
        gr.update(interactive=True),
        "",
    ]


def public_rows(snapshot: GradioSnapshot | None) -> list[list[Any]]:
    if snapshot is None:
        return []
    include_reason = not result_has_human(snapshot.result)
    rows = []
    for event in snapshot.result.get("public_history", [])[-120:]:
        _actor, message, _meta = conversation_message(event, include_reason=include_reason)
        rows.append(
            [
                event.get("round_id", ""),
                event.get("type", ""),
                event.get("player")
                or event.get("challenger")
                or event.get("shooter")
                or event.get("starting_player", ""),
                message,
            ]
        )
    return rows


def build_app():
    try:
        kbench = load_kbench()
        choices = model_choices(kbench)
        load_status = f"Loaded {len(choices)} kbench models."
    except Exception as exc:
        kbench = None
        choices = list(DEFAULT_MODEL_IDS)
        load_status = f"kbench is not loaded: {exc}"

    initial = default_state(choices)
    initial_snapshot = None
    css = """
    .lb-board {
      min-height:650px;
      background:#090705;
      border:1px solid #2f261f;
      border-radius:8px;
      padding:18px;
      color:#e5e7eb;
      overflow:hidden;
    }
    .lb-board.empty { display:flex; align-items:center; justify-content:center; }
    .lb-empty { color:#cbd5e1; font-size:18px; }
    .lb-table-wrap {
      min-height:560px;
      position:relative;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:122px 28px 96px;
      box-sizing:border-box;
      background:
        radial-gradient(circle at 15% 20%, rgba(245,158,11,.12), transparent 16%),
        radial-gradient(circle at 80% 70%, rgba(127,29,29,.18), transparent 18%),
        linear-gradient(120deg, rgba(255,255,255,.04), transparent 22% 76%, rgba(255,255,255,.03)),
        #0d0906;
      border-radius:8px;
    }
    .lb-table-felt {
      position:relative;
      width:min(90%, 960px);
      aspect-ratio: 2.15 / 1;
      min-height:330px;
      border-radius:26px;
      background:
        linear-gradient(90deg, rgba(255,255,255,.035), transparent 16% 22%, rgba(255,255,255,.028) 28% 31%, transparent 38% 42%, rgba(255,255,255,.03) 54% 58%, transparent 70%),
        radial-gradient(circle at center, rgba(245,158,11,.12), transparent 52%),
        linear-gradient(135deg, #3b2416, #22130b 46%, #140d09);
      border:12px solid #5a3925;
      box-shadow:
        0 0 0 5px #16100c,
        0 30px 52px rgba(0,0,0,.62),
        inset 0 0 0 1px rgba(245,158,11,.22),
        inset 0 0 68px rgba(0,0,0,.48);
      animation: lb-table-breathe 7s ease-in-out infinite;
    }
    .lb-table-felt::before {
      content:"";
      position:absolute;
      inset:24px;
      border:1px solid rgba(245,158,11,.23);
      border-radius:18px;
      pointer-events:none;
    }
    .lb-player-seat {
      position:absolute;
      width:174px;
      min-height:92px;
      transform:translate(-50%, -50%);
      background:#17110d;
      border:1px solid #493326;
      border-radius:8px;
      padding:30px 10px 9px;
      text-align:center;
      color:#e5e7eb;
      box-shadow:0 16px 28px rgba(0,0,0,.35);
      z-index:3;
    }
    .lb-player-seat.evaluated {
      border-color:#d97706;
      box-shadow:0 0 0 2px rgba(217,119,6,.4) inset, 0 16px 28px rgba(0,0,0,.35);
    }
    .lb-player-seat.active {
      border-color:#facc15;
      box-shadow:0 0 0 2px rgba(250,204,21,.45) inset, 0 16px 28px rgba(0,0,0,.35);
      animation: lb-seat-pulse 1.6s ease-in-out infinite;
    }
    .lb-player-seat.out {
      opacity:.62;
      filter:grayscale(.55);
      border-color:#7f1d1d;
      background:#120b09;
    }
    .lb-player-seat.out .lb-hand-stack { opacity:.25; }
    .lb-dead-stamp {
      position:absolute;
      inset:8px 8px auto auto;
      transform:rotate(6deg);
      border:2px solid #ef4444;
      color:#fecaca !important;
      background:rgba(127,29,29,.28);
      border-radius:5px;
      padding:2px 7px;
      font-size:11px;
      font-weight:950;
      letter-spacing:.08em;
      z-index:4;
    }
    .lb-player-seat.shot-safe {
      border-color:#f59e0b;
      animation: lb-seat-shot-safe .78s ease-out both;
    }
    .lb-player-seat.shot-hit {
      border-color:#ef4444;
      animation: lb-seat-shot-hit .9s ease-out both;
    }
    .seat-2-1 { left:50%; top:0%; }
    .seat-2-2 { left:50%; top:100%; }
    .seat-3-1 { left:50%; top:0%; }
    .seat-3-2 { left:7%; top:73%; }
    .seat-3-3 { left:93%; top:73%; }
    .seat-4-1 { left:50%; top:0%; }
    .seat-4-2 { left:93%; top:50%; }
    .seat-4-3 { left:50%; top:100%; }
    .seat-4-4 { left:7%; top:50%; }
    .lb-name {
      font-size:18px;
      font-weight:800;
      color:#f8fafc !important;
      line-height:1.15;
      overflow-wrap:anywhere;
    }
    .lb-model {
      margin-top:3px;
      color:#c8a879 !important;
      font-size:11px;
      line-height:1.2;
      overflow-wrap:anywhere;
    }
    .lb-seat-stats { display:flex; justify-content:center; gap:5px; margin-top:8px; flex-wrap:wrap; }
    .lb-counts { display:flex; gap:8px; align-items:flex-start; flex-wrap:wrap; justify-content:flex-end; }
    .lb-counts span, .lb-seat-stats span {
      background:#241a13;
      border:1px solid #5a3925;
      border-radius:6px;
      padding:4px 7px;
      color:#f8fafc !important;
      font-size:12px;
    }
    .lb-stat {
      min-width:28px;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:3px;
      font-weight:850;
      line-height:1;
      white-space:nowrap;
    }
    .lb-stat::before {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      width:13px;
      height:13px;
      border-radius:999px;
      font-size:9px;
      font-weight:950;
      color:#0f172a;
      background:#e5e7eb;
    }
    .lb-stat-life::before { content:"A"; background:#86efac; }
    .lb-player-seat.out .lb-stat-life::before { content:"X"; background:#fca5a5; }
    .lb-stat-shot::before { content:"R"; background:#fbbf24; }
    .lb-stat-card::before { content:"C"; background:#bfdbfe; }
    .lb-hand-stack {
      position:absolute;
      top:-34px;
      left:50%;
      transform:translateX(-50%);
      height:54px;
      display:flex;
      justify-content:center;
      min-width:88px;
    }
    .lb-hand-stack .lb-card { margin-left:-18px; }
    .lb-hand-stack .lb-card:first-child { margin-left:0; }
    .lb-hand-stack.face-up {
      top:-38px;
      gap:3px;
      min-width:164px;
    }
    .lb-hand-stack.face-up .lb-card { margin-left:0; }
    .lb-hand-stack.face-up .lb-card.small {
      width:30px;
      height:42px;
      border-radius:6px;
    }
    .lb-hand-stack.face-up .lb-card.small .rank { font-size:13px; }
    .lb-card {
      display:inline-flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      position:relative;
      overflow:hidden;
      width:44px;
      height:62px;
      border-radius:7px;
      background:
        radial-gradient(circle at 50% 28%, rgba(255,255,255,.12), transparent 28%),
        linear-gradient(145deg, #222832, #05070a 62%, #151922);
      border:2px solid #3f4856;
      color:#f8fafc !important;
      font-weight:900;
      box-shadow:0 5px 12px rgba(0,0,0,.34);
      line-height:1;
      animation: lb-card-pop .24s ease-out both;
    }
    .lb-card.small { width:34px; height:48px; border-radius:6px; font-size:12px; }
    .lb-card .rank {
      position:relative;
      z-index:1;
      font-size:18px;
      color:currentColor !important;
      text-shadow:0 1px 5px rgba(0,0,0,.65);
    }
    .lb-card.small .rank { font-size:13px; }
    .lb-card.rank-q {
      color:#fda4af !important;
      border-color:#fb7185;
      background:
        radial-gradient(circle at 50% 28%, rgba(251,113,133,.24), transparent 30%),
        linear-gradient(145deg, #32131a, #080508 62%, #1f0a10);
    }
    .lb-card.rank-k {
      color:#93c5fd !important;
      border-color:#60a5fa;
      background:
        radial-gradient(circle at 50% 28%, rgba(96,165,250,.22), transparent 30%),
        linear-gradient(145deg, #12233a, #05070b 62%, #0d1725);
    }
    .lb-card.rank-a {
      color:#fde68a !important;
      border-color:#f59e0b;
      background:
        radial-gradient(circle at 50% 28%, rgba(245,158,11,.25), transparent 30%),
        linear-gradient(145deg, #332308, #070604 62%, #1f1606);
    }
    .lb-card.rank-joker {
      color:#d8b4fe !important;
      border-color:#a855f7;
      background:
        radial-gradient(circle at 50% 28%, rgba(168,85,247,.26), transparent 30%),
        linear-gradient(145deg, #26143d, #06040a 62%, #1a0d2b);
    }
    .lb-card.back {
      background:
        linear-gradient(45deg, transparent 0 18%, rgba(255,255,255,.22) 18% 25%, transparent 25% 45%, rgba(0,0,0,.24) 45% 51%, transparent 51% 72%, rgba(255,255,255,.16) 72% 78%, transparent 78%),
        linear-gradient(-45deg, transparent 0 14%, rgba(0,0,0,.18) 14% 20%, transparent 20% 39%, rgba(255,255,255,.14) 39% 45%, transparent 45% 66%, rgba(0,0,0,.2) 66% 72%, transparent 72%),
        radial-gradient(circle at 38% 28%, rgba(255,255,255,.12), transparent 24%),
        #7f1d1d;
      background-size:17px 17px, 23px 23px, 100% 100%, 100% 100%;
      background-position:0 0, 7px 4px, 0 0, 0 0;
      border-color:#f59e0b;
      color:transparent !important;
    }
    .lb-card.back .rank { color:transparent !important; }
    .lb-pot {
      position:absolute;
      left:50%;
      top:50%;
      transform:translate(-50%, -50%);
      width:min(58%, 390px);
      min-height:220px;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      gap:8px;
      text-align:center;
      color:#f8fafc;
      z-index:2;
    }
    .lb-round {
      color:#fef3c7 !important;
      font-size:16px;
      font-weight:800;
      text-transform:uppercase;
    }
    .lb-target { display:flex; align-items:center; gap:8px; color:#e5e7eb !important; }
    .lb-target span { color:#e5e7eb !important; font-size:13px; font-weight:800; }
    .lb-pot p {
      margin:0;
      color:#f8fafc !important;
      font-weight:700;
      line-height:1.15;
      font-size:13px;
      overflow-wrap:anywhere;
    }
    .lb-pot p.challenge { color:#fde68a !important; animation: lb-warning-pulse 1.6s ease-in-out infinite; }
    .lb-pot-cards { display:flex; gap:5px; min-height:64px; align-items:center; justify-content:center; }
    .lb-pot-cards .lb-card { animation: lb-card-slide .32s ease-out both; }
    .lb-pot-cards.muted { color:#cbd5e1 !important; font-weight:700; min-height:30px; }
    .lb-revealed {
      display:flex;
      align-items:center;
      justify-content:center;
      gap:5px;
      color:#f8fafc !important;
      font-weight:800;
      flex-wrap:wrap;
    }
    .lb-revealed span { margin-right:3px; color:#f8fafc !important; }
    .lb-revealed .lb-card { animation: lb-reveal-card .34s ease-out both; }
    .lb-shot-overlay {
      position:absolute;
      inset:0;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      gap:7px;
      pointer-events:none;
      z-index:5;
      border-radius:18px;
      background:radial-gradient(circle, rgba(0,0,0,.48), rgba(0,0,0,.05) 58%, transparent 70%);
      animation: lb-shot-overlay-out 1.25s ease-out both;
    }
    .lb-shot-overlay.hit {
      background:radial-gradient(circle, rgba(127,29,29,.62), rgba(0,0,0,.12) 58%, transparent 72%);
    }
    .lb-revolver {
      position:relative;
      width:68px;
      height:68px;
      border-radius:999px;
      border:5px solid #d6d3d1;
      background:#1c1917;
      box-shadow:0 0 22px rgba(250,204,21,.45);
      animation: lb-cylinder-spin .55s ease-out both;
    }
    .lb-revolver span {
      position:absolute;
      width:12px;
      height:12px;
      border-radius:999px;
      background:#57534e;
      left:50%;
      top:50%;
      transform-origin:0 0;
    }
    .lb-revolver span:nth-child(1) { transform:rotate(0deg) translate(19px, -6px); }
    .lb-revolver span:nth-child(2) { transform:rotate(60deg) translate(19px, -6px); }
    .lb-revolver span:nth-child(3) { transform:rotate(120deg) translate(19px, -6px); }
    .lb-revolver span:nth-child(4) { transform:rotate(180deg) translate(19px, -6px); }
    .lb-revolver span:nth-child(5) { transform:rotate(240deg) translate(19px, -6px); }
    .lb-revolver span:nth-child(6) { transform:rotate(300deg) translate(19px, -6px); }
    .lb-shot-word {
      color:#fef3c7 !important;
      font-size:32px;
      font-weight:950;
      letter-spacing:.08em;
      text-shadow:0 0 18px rgba(245,158,11,.55);
      animation: lb-shot-word .58s ease-out both;
    }
    .lb-shot-overlay.hit .lb-shot-word {
      color:#fecaca !important;
      text-shadow:0 0 22px rgba(239,68,68,.8);
    }
    .lb-shot-progress {
      color:#fde68a !important;
      border:1px solid rgba(245,158,11,.65);
      background:rgba(28,25,23,.78);
      border-radius:6px;
      padding:3px 9px;
      font-weight:900;
      letter-spacing:.05em;
    }
    .lb-shot-subtext {
      color:#f8fafc !important;
      font-size:13px;
      font-weight:800;
      text-shadow:0 2px 8px #000;
    }
    .lb-table-footer {
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      margin-top:8px;
      border-top:1px solid #243244;
      padding-top:8px;
      min-height:36px;
      color:#e5e7eb;
    }
    .lb-footer-main {
      min-width:0;
      display:flex;
      align-items:baseline;
      gap:8px;
      flex-wrap:wrap;
    }
    .lb-table-footer h2 {
      margin:0;
      font-size:14px;
      line-height:1.1;
      color:#f8fafc !important;
      overflow-wrap:anywhere;
    }
    .lb-table-footer p {
      margin:0;
      color:#cbd5e1 !important;
      font-size:10px;
      line-height:1.1;
      white-space:nowrap;
    }
    .lb-counts { display:flex; gap:5px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .lb-table-footer .lb-counts span {
      font-size:10px;
      padding:3px 6px;
      line-height:1;
      min-width:auto;
    }
    .lb-conversation {
      max-height:560px;
      overflow:auto;
      background:#101722;
      border:1px solid #263449;
      border-radius:8px;
      padding:12px;
      color:#e5e7eb;
      margin-bottom:12px;
    }
    .lb-conversation-title {
      text-transform:uppercase;
      letter-spacing:.06em;
      color:#94a3b8 !important;
      font-size:12px;
      font-weight:850;
      margin-bottom:8px;
    }
    .lb-chat-bubble {
      display:grid;
      grid-template-columns:14px 1fr;
      gap:10px;
      padding:10px;
      margin:8px 0;
      background:#182233;
      border:1px solid #263449;
      border-radius:8px;
    }
    .lb-chat-bubble.system {
      background:#111827;
      border-color:#334155;
    }
    .lb-chat-dot {
      width:12px;
      height:12px;
      border-radius:999px;
      margin-top:4px;
      border:1px solid #f8fafc;
    }
    .lb-chat-head {
      display:flex;
      justify-content:space-between;
      gap:8px;
      align-items:baseline;
      margin-bottom:4px;
    }
    .lb-chat-head span { color:#f8fafc !important; font-weight:850; }
    .lb-chat-head small { color:#94a3b8 !important; font-size:11px; white-space:nowrap; }
    .lb-chat-message {
      color:#e5e7eb !important;
      line-height:1.5;
      overflow-wrap:anywhere;
    }
    .lb-empty-chat {
      color:#cbd5e1 !important;
      padding:8px 0;
    }
    .lb-side {
      min-height:180px;
      background:#111827;
      border:1px solid #475569;
      border-radius:8px;
      padding:14px;
      color:#e5e7eb;
    }
    .lb-side h3 { color:#f8fafc !important; }
    .lb-side p, .lb-side b, .lb-side pre { color:#e5e7eb !important; }
    .lb-human-panel {
      background:#111827;
      border:1px solid #f59e0b;
      border-radius:8px;
      padding:10px;
      margin:0 0 12px;
      color:#f8fafc;
    }
    .lb-human-prompt h3 { margin:0 0 6px; color:#fef3c7 !important; }
    .lb-human-prompt p { margin:0; color:#e5e7eb !important; line-height:1.35; font-size:12px; }
    .lb-human-panel label, .lb-human-panel span, .lb-human-panel textarea {
      font-size:12px !important;
    }
    @keyframes lb-table-breathe {
      0%, 100% { box-shadow:0 0 0 5px #16100c, 0 30px 52px rgba(0,0,0,.62), inset 0 0 0 1px rgba(245,158,11,.2), inset 0 0 68px rgba(0,0,0,.48); }
      50% { box-shadow:0 0 0 5px #16100c, 0 32px 58px rgba(0,0,0,.68), inset 0 0 0 1px rgba(245,158,11,.34), inset 0 0 78px rgba(0,0,0,.52); }
    }
    @keyframes lb-seat-pulse {
      0%, 100% { transform:translate(-50%, -50%) scale(1); }
      50% { transform:translate(-50%, -50%) scale(1.025); }
    }
    @keyframes lb-card-pop {
      from { opacity:0; transform:translateY(5px) scale(.96); }
      to { opacity:1; transform:translateY(0) scale(1); }
    }
    @keyframes lb-card-slide {
      from { opacity:0; transform:translateY(-9px) rotate(-2deg); }
      to { opacity:1; transform:translateY(0) rotate(0); }
    }
    @keyframes lb-reveal-card {
      from { opacity:0; transform:rotateY(70deg) scale(.92); }
      to { opacity:1; transform:rotateY(0) scale(1); }
    }
    @keyframes lb-warning-pulse {
      0%, 100% { color:#fde68a; }
      50% { color:#fca5a5; }
    }
    @keyframes lb-cylinder-spin {
      from { transform:rotate(-160deg) scale(.86); opacity:.2; }
      65% { transform:rotate(18deg) scale(1.06); opacity:1; }
      to { transform:rotate(0deg) scale(1); opacity:1; }
    }
    @keyframes lb-shot-word {
      from { transform:scale(.62); opacity:0; }
      50% { transform:scale(1.16); opacity:1; }
      to { transform:scale(1); opacity:1; }
    }
    @keyframes lb-shot-overlay-out {
      0%, 74% { opacity:1; }
      100% { opacity:.14; }
    }
    @keyframes lb-seat-shot-safe {
      0% { transform:translate(-50%, -50%) rotate(0deg); }
      22% { transform:translate(-50%, -50%) rotate(-2deg); }
      48% { transform:translate(-50%, -50%) rotate(2deg); }
      100% { transform:translate(-50%, -50%) rotate(0deg); }
    }
    @keyframes lb-seat-shot-hit {
      0% { transform:translate(-50%, -50%) scale(1); filter:none; }
      38% { transform:translate(-50%, -50%) scale(1.05); filter:drop-shadow(0 0 18px rgba(239,68,68,.9)); }
      100% { transform:translate(-50%, -50%) scale(.98); filter:grayscale(.6); }
    }
    @media (max-width: 760px) {
      .lb-board { padding:10px; min-height:720px; }
      .lb-table-wrap { min-height:500px; padding:108px 12px 90px; }
      .lb-table-felt { width:96%; min-height:300px; border-width:14px; }
      .lb-player-seat { width:142px; min-height:92px; padding:28px 8px 8px; }
      .lb-name { font-size:14px; }
      .lb-seat-stats span { font-size:10px; padding:3px 5px; }
      .lb-pot { width:54%; min-height:170px; gap:5px; }
      .lb-pot p { font-size:11px; }
      .lb-round { font-size:12px; }
      .lb-card { width:34px; height:48px; }
      .lb-card .rank { font-size:15px; }
      .lb-card.small { width:26px; height:37px; }
      .lb-table-footer { flex-direction:row; align-items:flex-start; }
      .lb-footer-main { gap:5px; }
      .lb-table-footer h2 { font-size:12px; }
      .lb-table-footer p { font-size:9px; }
      .seat-2-1, .seat-3-1, .seat-4-1 { top:0%; }
      .seat-2-2, .seat-4-3 { top:100%; }
      .seat-3-2, .seat-4-4 { left:4%; }
      .seat-3-3, .seat-4-2 { left:96%; }
      .lb-conversation { max-height:360px; }
    }
    @media (prefers-color-scheme: light) {
      .lb-board {
        background:#e8edf4;
        border-color:#cbd5e1;
        color:#111827;
      }
      .lb-table-wrap { background:#d7dee8; }
      .lb-player-seat {
        background:#ffffff;
        border-color:#cbd5e1;
        color:#111827;
      }
      .lb-name { color:#111827 !important; }
      .lb-model { color:#475569 !important; }
      .lb-counts span, .lb-seat-stats span {
        background:#e5e7eb;
        border-color:#cbd5e1;
        color:#111827 !important;
      }
      .lb-table-footer { border-color:#cbd5e1; color:#111827; }
      .lb-table-footer h2 { color:#111827 !important; }
      .lb-table-footer p { color:#475569 !important; }
      .lb-conversation { background:#ffffff; border-color:#cbd5e1; color:#1f2937; }
      .lb-conversation-title { color:#64748b !important; }
      .lb-chat-bubble { background:#f8fafc; border-color:#d1d5db; }
      .lb-chat-bubble.system { background:#f1f5f9; border-color:#cbd5e1; }
      .lb-chat-dot { border-color:#111827; }
      .lb-chat-head span { color:#111827 !important; }
      .lb-chat-head small { color:#64748b !important; }
      .lb-chat-message { color:#1f2937 !important; }
      .lb-empty-chat { color:#64748b !important; }
      .lb-side { background:#ffffff; border-color:#d1d5db; color:#1f2937; }
      .lb-side h3 { color:#111827 !important; }
      .lb-side p, .lb-side b, .lb-side pre { color:#1f2937 !important; }
      .lb-human-panel { background:#ffffff; border-color:#d97706; color:#111827; }
      .lb-human-prompt h3 { color:#92400e !important; }
      .lb-human-prompt p { color:#1f2937 !important; }
    }
    """

    with gr.Blocks(title="kbench_liarsbar") as demo:
        gr.HTML(f"<style>{css}</style>")
        state = gr.State(initial)
        latest_snapshot = gr.State(initial_snapshot)

        with gr.Row():
            with gr.Column(scale=7):
                game_html = gr.HTML(empty_display())
            with gr.Column(scale=3):
                with gr.Group(visible=False, elem_classes=["lb-human-panel"]) as human_panel:
                    human_prompt = gr.HTML("")
                    human_cards = gr.CheckboxGroup(label="Choose cards", choices=[], visible=False)
                    human_text = gr.Textbox(
                        label="Public table message",
                        lines=3,
                        placeholder="Write what this player says or does publicly at the table.",
                        interactive=False,
                    )
                    human_challenge = gr.Radio(
                        choices=[
                            ("Do not challenge", "no_challenge"),
                            ("Challenge", "challenge"),
                        ],
                        value="no_challenge",
                        label="Challenge decision",
                        visible=False,
                    )
                    human_submit = gr.Button("Submit Human Action", variant="primary", interactive=False)
                    human_error = gr.Markdown("")
                side_html = gr.HTML(render_side(None))

        with gr.Group(visible=True) as config_scene:
            gr.Markdown("## Game Config")
            gr.Markdown(load_status)
            with gr.Row():
                with gr.Column(scale=2):
                    player_count = gr.Slider(2, 4, value=initial["player_count"], step=1, label="Number of players")
                    with gr.Row():
                        revolver_chambers = gr.Number(
                            value=initial["revolver_chambers"],
                            precision=0,
                            label="Revolver chambers",
                        )
                        enable_reflection = gr.Checkbox(
                            value=initial["enable_reflection"],
                            label="Reflection memory",
                        )
                    randomize_btn = gr.Button("Randomize Names")
                with gr.Column(scale=3):
                    player_rows_ui = []
                    name_inputs = []
                    model_inputs = []
                    personality_inputs = []
                    for index in range(MAX_PLAYERS):
                        with gr.Group(visible=index < initial["player_count"]) as row:
                            with gr.Row():
                                name = gr.Textbox(initial["players"][index]["name"], label=f"Seat {index + 1} name")
                                model = gr.Dropdown(
                                    choices,
                                    value=initial["players"][index]["model"],
                                    label=f"Seat {index + 1} model",
                                    allow_custom_value=True,
                                )
                            personality = gr.Textbox(
                                value=initial["players"][index].get("personality", ""),
                                label=f"Seat {index + 1} custom prompt",
                                placeholder="Optional strategy, personality, or table-talk style for this player.",
                                lines=2,
                            )
                        player_rows_ui.append(row)
                        name_inputs.append(name)
                        model_inputs.append(model)
                        personality_inputs.append(personality)
            validation = gr.Markdown("")
            with gr.Row():
                validate_btn = gr.Button("Validate Config")
                export_btn = gr.Button("Export Config")
                play_btn = gr.Button("Play", variant="primary")
            export_box = gr.Code(
                label="Copyable Python config",
                language="python",
                lines=22,
                value="# Click Export Config to generate a copyable snippet.",
            )

        with gr.Group(visible=False) as gameplay_scene:
            gr.Markdown("## Gameplay")
            with gr.Row():
                stop_btn = gr.Button("Stop", variant="stop", interactive=False)
                restart_btn = gr.Button("Back to Config", interactive=False)
            public_log = gr.Dataframe(
                headers=["Round", "Type", "Actor", "Summary"],
                datatype=["number", "str", "str", "str"],
                label="Public Log",
                visible=False,
            )
            result_json = gr.JSON(label="Result JSON")

        collect_inputs = [
            player_count,
            revolver_chambers,
            enable_reflection,
        ]
        for index in range(MAX_PLAYERS):
            collect_inputs.extend(
                [
                    name_inputs[index],
                    model_inputs[index],
                    personality_inputs[index],
                ]
            )

        player_count.change(
            update_player_visibility,
            inputs=[player_count],
            outputs=player_rows_ui,
        )
        randomize_btn.click(
            randomize_names,
            inputs=[player_count],
            outputs=name_inputs,
        )

        def validate_only(*values):
            new_state = collect_state(*values)
            try:
                validate_state(new_state)
            except Exception as exc:
                return new_state, f"### Validation failed\n```text\n{exc}\n```"
            return new_state, "### Config valid"

        validate_btn.click(
            validate_only,
            inputs=collect_inputs,
            outputs=[state, validation],
        )

        def export_only(*values):
            new_state = collect_state(*values)
            try:
                code = export_config_code(new_state)
            except Exception as exc:
                return new_state, f"### Export failed\n```text\n{exc}\n```", ""
            return new_state, "### Export ready", code

        export_btn.click(
            export_only,
            inputs=collect_inputs,
            outputs=[state, validation, export_box],
        )

        def run_game_stream(*values):
            if kbench is None:
                missing_state = collect_state(*values)
                has_human = state_has_human(missing_state)
                yield (
                    missing_state,
                    None,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    empty_display("kbench is not loaded."),
                    render_side_notice(None, "Error", "Check kaggle-benchmarks/.env."),
                    [],
                    result_json_update({}, has_human),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    *hidden_human_controls(),
                )
                return
            new_state = collect_state(*values)
            has_human = state_has_human(new_state)
            run_id = str(time.time_ns())
            stop_event = threading.Event()
            RUN_STOP_EVENTS[run_id] = stop_event
            if has_human:
                RUN_HUMAN_INPUTS[run_id] = queue.Queue()
            new_state["run_id"] = run_id
            try:
                game_config = make_game_config(kbench, new_state)
            except Exception as exc:
                RUN_STOP_EVENTS.pop(run_id, None)
                RUN_HUMAN_INPUTS.pop(run_id, None)
                yield (
                    new_state,
                    None,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    empty_display("Validation failed."),
                    render_side_notice(None, "Validation failed", str(exc)),
                    [],
                    result_json_update({}, has_human),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    *hidden_human_controls(),
                )
                return

            updates: queue.Queue[Any] = queue.Queue()
            ui = GradioGameUI(updates, stop_event)
            result_box: dict[str, Any] = {}

            def target():
                try:
                    game = LiarsBarGame(game_config=game_config, UI=ui)
                    result_box["result"] = game.start()
                except GameStopped:
                    result_box["stopped"] = True
                except Exception as exc:
                    result_box["error"] = exc
                finally:
                    updates.put(None)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            last_snapshot = None
            yield (
                new_state,
                None,
                gr.update(visible=False),
                gr.update(visible=True),
                empty_display("Game starting..."),
                render_side_notice(
                    None,
                    "Running",
                    "Game is starting. Stop is available after this screen appears.",
                ),
                [],
                result_json_update({}, has_human),
                gr.update(interactive=True),
                gr.update(interactive=False),
                *hidden_human_controls(),
            )
            while True:
                item = updates.get()
                if item is None:
                    break
                if isinstance(item, HumanRequest):
                    new_state["human_request"] = item.to_payload()
                    yield (
                        new_state,
                        last_snapshot,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        render_game(last_snapshot),
                        render_side(last_snapshot),
                        public_rows(last_snapshot),
                        result_json_update(last_snapshot.result if last_snapshot else {}, has_human),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        *visible_human_controls(item),
                    )
                    continue
                last_snapshot = item
                stop_requested = stop_event.is_set()
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_game(last_snapshot),
                    render_side(last_snapshot),
                    public_rows(last_snapshot),
                    result_json_update(last_snapshot.result, has_human),
                    gr.update(interactive=not stop_requested),
                    gr.update(interactive=False),
                    *hidden_human_controls(),
                )
            RUN_STOP_EVENTS.pop(run_id, None)
            RUN_HUMAN_INPUTS.pop(run_id, None)
            if result_box.get("stopped"):
                if last_snapshot is not None:
                    last_snapshot.report_text = "Game stopped by user."
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_game(last_snapshot),
                    render_side_notice(
                        last_snapshot,
                        "Stopped",
                        "Game stopped by user. In-flight LLM calls may finish, but no new calls are started after the stop signal.",
                    ),
                    public_rows(last_snapshot),
                    result_json_update(last_snapshot.result if last_snapshot else {}, has_human),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    *hidden_human_controls(),
                )
                return
            if "error" in result_box:
                error = result_box["error"]
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_game(last_snapshot),
                    render_side_notice(last_snapshot, "Game failed", str(error)),
                    public_rows(last_snapshot),
                    result_json_update(last_snapshot.result if last_snapshot else {}, has_human),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    *hidden_human_controls(),
                )
                return
            final_result = result_box.get("result", {})
            new_state["game_log"] = final_result.get("game_log")
            final_snapshot = GradioSnapshot(report_text=final_result.get("winner") or "Game ended.", result=final_result)
            yield (
                new_state,
                final_snapshot,
                gr.update(visible=False),
                gr.update(visible=True),
                render_game(final_snapshot),
                render_side(final_snapshot),
                public_rows(final_snapshot),
                result_json_update(final_result, has_human),
                gr.update(interactive=False),
                gr.update(interactive=True),
                *hidden_human_controls(),
            )

        play_btn.click(
            run_game_stream,
            inputs=collect_inputs,
            outputs=[
                state,
                latest_snapshot,
                config_scene,
                gameplay_scene,
                game_html,
                side_html,
                public_log,
                result_json,
                stop_btn,
                restart_btn,
                human_panel,
                human_prompt,
                human_cards,
                human_text,
                human_challenge,
                human_submit,
                human_error,
            ],
        )

        def stop_game(state_value, snapshot_value):
            state_value = state_value or {}
            run_id = state_value.get("run_id")
            state_value["stop_requested"] = True
            if run_id and run_id in RUN_STOP_EVENTS:
                RUN_STOP_EVENTS[run_id].set()
            if snapshot_value is not None:
                snapshot_value.report_text = "Stop requested. Waiting for the current LLM/action boundary."
            return (
                state_value,
                snapshot_value,
                render_game(snapshot_value),
                render_side_notice(
                    snapshot_value,
                    "Stop requested",
                    "Waiting for the current LLM/action boundary. New LLM calls are blocked after the stop signal.",
                ),
                public_rows(snapshot_value),
                gr.update(interactive=False),
                gr.update(interactive=False),
                *hidden_human_controls(),
            )

        stop_btn.click(
            stop_game,
            inputs=[state, latest_snapshot],
            outputs=[
                state,
                latest_snapshot,
                game_html,
                side_html,
                public_log,
                stop_btn,
                restart_btn,
                human_panel,
                human_prompt,
                human_cards,
                human_text,
                human_challenge,
                human_submit,
                human_error,
            ],
        )

        def submit_human_action(state_value, selected_cards, public_text, challenge_choice):
            state_value = state_value or {}
            request = state_value.get("human_request") or {}
            run_id = request.get("run_id") or state_value.get("run_id")
            inputs = RUN_HUMAN_INPUTS.get(str(run_id))
            if not request or inputs is None:
                return state_value, "No pending human decision.", gr.update(interactive=False)
            message = str(public_text or "").strip()
            phase = request.get("phase")
            if phase == "play":
                selected = list(selected_cards or [])
                legal_counts = set(int(value) for value in request.get("legal_card_counts", []))
                if len(selected) not in legal_counts:
                    return (
                        state_value,
                        f"Choose one of these legal card counts: {sorted(legal_counts)}.",
                        gr.update(interactive=True),
                    )
                hand = list(request.get("hand", []))
                played_cards = []
                try:
                    for item in selected:
                        index = int(str(item).split(":", 1)[0]) - 1
                        played_cards.append(hand[index])
                except Exception:
                    return state_value, "Card selection could not be parsed.", gr.update(interactive=True)
                inputs.put(
                    {
                        "phase": "play",
                        "player_name": request.get("player_name"),
                        "played_cards": played_cards,
                        "behavior": message,
                    }
                )
            elif phase == "challenge":
                inputs.put(
                    {
                        "phase": "challenge",
                        "player_name": request.get("player_name"),
                        "was_challenged": challenge_choice == "challenge",
                        "challenge_reason": message,
                    }
                )
            else:
                return state_value, "Unknown human decision phase.", gr.update(interactive=False)
            state_value["human_request"] = None
            return state_value, "Submitted. Waiting for the game to continue.", gr.update(interactive=False)

        human_submit.click(
            submit_human_action,
            inputs=[state, human_cards, human_text, human_challenge],
            outputs=[state, human_error, human_submit],
        )

        def restart(state_value):
            return (
                state_value or initial,
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                empty_display(),
                render_side(None),
                [],
                {},
                gr.update(interactive=False),
                gr.update(interactive=False),
                *hidden_human_controls(),
            )

        restart_btn.click(
            restart,
            inputs=[state],
            outputs=[
                state,
                latest_snapshot,
                config_scene,
                gameplay_scene,
                game_html,
                side_html,
                public_log,
                result_json,
                stop_btn,
                restart_btn,
                human_panel,
                human_prompt,
                human_cards,
                human_text,
                human_challenge,
                human_submit,
                human_error,
            ],
        )

    return demo


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Launch the kbench_liarsbar Gradio app.")
    parser.add_argument("--share", default=False)
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args(argv)

    share = str(args.share).lower() in {"1", "true", "yes", "y"}
    app = build_app()
    try:
        app.launch(
            share=share,
            server_name=args.server_name,
            server_port=args.server_port,
        )
    except KeyboardInterrupt:
        for stop_event in list(RUN_STOP_EVENTS.values()):
            stop_event.set()
        raise
    finally:
        for stop_event in list(RUN_STOP_EVENTS.values()):
            stop_event.set()


if __name__ == "__main__":
    main()
