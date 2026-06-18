from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from faker import Faker


@dataclass
class GameConfig:
    """Experiment config for an explicit-player Liars Bar run."""

    player_configs: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None
    max_rounds: int = 30
    max_turns: int = 240
    revolver_chambers: int = 6
    enable_reflection: bool = True
    evaluated_player_name: str = "Evaluated"
    opponent_model_ids: list[str] = field(default_factory=list)

    def with_updates(self, **updates) -> "GameConfig":
        data = {
            "player_configs": [dict(spec) for spec in self.player_configs],
            "seed": self.seed,
            "max_rounds": self.max_rounds,
            "max_turns": self.max_turns,
            "revolver_chambers": self.revolver_chambers,
            "enable_reflection": self.enable_reflection,
            "evaluated_player_name": self.evaluated_player_name,
            "opponent_model_ids": list(self.opponent_model_ids),
        }
        data.update(updates)
        return GameConfig(**data)


def build_benchmark_config(
    kbench,
    evaluated_llm,
    *,
    opponent_model_ids: list[str],
    seed: int | None = None,
    revolver_chambers: int = 6,
    player_names: list[str] | None = None,
) -> GameConfig:
    """Build the default mixed-LLM benchmark configuration.

    The evaluated model is seat 1. Opponents are resolved from provider-qualified
    `kbench.llms` keys supplied by the task author. The task may provide one
    to three opponents, producing a two- to four-player game. Missing or
    out-of-range opponent lists raise setup errors instead of falling back to
    deterministic agents.
    """

    if not 1 <= len(opponent_model_ids) <= 3:
        raise ValueError(
            "Liars Bar benchmark tasks must define 1 to 3 opponent models "
            f"for a 2- to 4-player game; got {len(opponent_model_ids)}."
        )
    available = dict(getattr(kbench, "llms", {}) or {})
    selected = _select_opponent_models(
        available,
        evaluated_llm,
        opponent_model_ids,
    )
    resolved_names = player_names or generate_player_names(len(selected) + 1, seed=seed)
    if len(resolved_names) != len(selected) + 1:
        raise ValueError(
            "player_names must contain exactly one name for the evaluated LLM "
            f"plus each opponent; expected {len(selected) + 1}, got "
            f"{len(resolved_names)}."
        )
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("player_names must be distinct.")
    player_configs = [
        {
            "name": resolved_names[0],
            "agent": evaluated_llm,
            "model_id": _model_name(evaluated_llm),
            "evaluated": True,
        }
    ]
    for index, (model_id, llm) in enumerate(selected, start=1):
        player_configs.append(
            {
                "name": resolved_names[index],
                "agent": llm,
                "model_id": model_id,
                "evaluated": False,
            }
        )
    return GameConfig(
        player_configs=player_configs,
        seed=seed,
        revolver_chambers=revolver_chambers,
        evaluated_player_name=resolved_names[0],
        opponent_model_ids=list(opponent_model_ids),
    )


def _select_opponent_models(available, evaluated_llm, opponent_model_ids):
    selected = []
    excluded_names = {_model_name(evaluated_llm)}

    for model_id in opponent_model_ids:
        llm = available.get(model_id)
        if llm is None:
            raise RuntimeError(
                f"Opponent model {model_id!r} is not available in kbench.llms."
            )
        if llm is evaluated_llm or model_id in excluded_names:
            raise RuntimeError(
                f"Opponent model {model_id!r} resolves to the evaluated model; "
                "choose distinct opponent models."
            )
        selected.append((model_id, llm))

    if len({model_id for model_id, _ in selected}) != len(selected):
        raise RuntimeError("Opponent model IDs must be distinct values.")
    return selected


def _model_name(llm) -> str:
    for attr in ("model", "name", "id"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return type(llm).__name__


def generate_player_names(count: int, *, seed: int | None = None) -> list[str]:
    if not 2 <= count <= 4:
        raise ValueError("Liars Bar player name generation expects 2 to 4 players.")
    faker = Faker()
    if seed is not None:
        faker.seed_instance(seed)
    names = []
    attempts = 0
    while len(names) < count and attempts < 100:
        attempts += 1
        name = faker.first_name()
        if name not in names:
            names.append(name)
    if len(names) != count:
        fallback = ["Ari", "Mika", "Noa", "Rin"]
        for name in fallback:
            if name not in names:
                names.append(name)
            if len(names) == count:
                break
    return names
