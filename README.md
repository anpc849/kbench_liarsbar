# kbench_liarsbar

Configurable **Liars Bar** environment for Kaggle Benchmarks LLM agents.

The environment owns the game rules: dealing cards, hidden information,
challenge resolution, revolver penalties, player elimination, replay logs, and
scoring. Agents only provide decisions.

## Installation

```bash
git clone https://github.com/anpc849/kbench_liarsbar
cd kbench_liarsbar
pip install -e .
```

## Gradio App

Launch the interactive Gradio app:

```bash
kbench_liarsbar_gradio --share True
```

The Gradio app expects `kaggle_benchmarks` to be importable. In Kaggle
notebooks, models are usually loaded automatically by the benchmark
environment. For local desktop testing, the app falls back to a local
`kaggle-benchmarks/.env` file and `kaggle-benchmarks/src` checkout when needed.

The UI supports:

- 2 to 4 players
- LLM players and optional human players
- custom or randomized player names
- custom extra prompt per player
- configurable revolver chambers
- optional reflection memory
- replay inspection and exportable config code

## Basic Usage

```python
from kbench_liarsbar import GameConfig, run_liarsbar_game

config = GameConfig(
    player_configs=[
        {"name": "Mira", "agent": agent_1, "model_id": "model-a", "evaluated": True},
        {"name": "Theo", "agent": agent_2, "model_id": "model-b"},
    ],
    revolver_chambers=3,
    seed=None,
)

result = run_liarsbar_game(game_config=config)
evaluated_won = result["winner"] == "Mira"
score = 1 if evaluated_won else 0
```

## Configuration

`GameConfig` controls one full game.

```python
GameConfig(
    player_configs=[...],
    seed=None,
    max_rounds=30,
    max_turns=240,
    revolver_chambers=6,
    enable_reflection=True,
    evaluated_player_name="Evaluated",
    opponent_model_ids=[],
)
```

### `player_configs`

Required. A list of 2 to 4 player specs.

Each player spec supports:

- `name`: public player name. Names must be distinct.
- `agent`: object that implements the Liars Bar agent methods, or a Kaggle
  Benchmarks LLM that can be wrapped by the default agent.
- `model_id`: optional public model label for UI and logs.
- `evaluated`: optional boolean. Mark the benchmarked player with `True`.
- `custom_prompt`: optional extra instruction appended to the default LLM
  prompt. It does not replace the rules prompt.

Example:

```python
player_configs = [
    {
        "name": "Mira",
        "agent": evaluated_llm,
        "model_id": "openai/gpt-5-mini",
        "evaluated": True,
        "custom_prompt": "Play cautiously and explain uncertainty clearly.",
    },
    {
        "name": "Theo",
        "agent": opponent_llm,
        "model_id": "google/gemini-3.5-flash",
    },
]
```

### `seed`

Optional. Use an integer for deterministic games, or `None` for fresh
randomization.

- `seed=123`: the same seed gives the same card distribution and random events
  across runs.
- `seed=None`: each run uses fresh randomness. Each round creates and shuffles a
  new deck.

### `max_rounds` and `max_turns`

Safety limits. The game stops if either limit is reached before a single winner
is found.

### `revolver_chambers`

Controls penalty-shot probability. The value must be at least `1`.

```python
GameConfig(..., revolver_chambers=6)  # 1 in 6 chance per shot
GameConfig(..., revolver_chambers=3)  # shorter game, 1 in 3 chance
GameConfig(..., revolver_chambers=1)  # every penalty shot eliminates
```

### `enable_reflection`

When `True`, LLM agents may update private reflection memory after public game
events. Reflection memory is private to the player and can store opinions about
other players. Set it to `False` for simpler, no-memory runs.

### `evaluated_player_name`

Name of the evaluated player. This is mainly used in result summaries and
benchmark scoring.

### `opponent_model_ids`

Metadata used by `build_benchmark_config()` to record which configured Kaggle
Benchmarks models were selected as opponents.

## Kaggle Benchmark Usage

`build_benchmark_config()` creates a 2 to 4 player game where the evaluated LLM
is the first player and the task author chooses 1 to 3 opponent models.

```python
import kaggle_benchmarks as kbench
import kbench_liarsbar as liarsbar

OPPONENT_MODEL_IDS = [
    "google/gemini-3.5-flash",
    "anthropic/claude-haiku-4-5@20251001",
    "openai/gpt-5-mini",
]


@kbench.task(name="kbench-liarsbar")
def kbench_liarsbar(llm) -> int:
    config = liarsbar.build_benchmark_config(
        kbench,
        llm,
        opponent_model_ids=OPPONENT_MODEL_IDS,
        player_names=["Mira", "Theo", "Nora", "Caleb"],
        revolver_chambers=3,
        seed=None,
    )
    result = liarsbar.run_liarsbar_game(game_config=config)
    return 1 if result["winner"] == config.evaluated_player_name else 0
```

`opponent_model_ids` must contain 1 to 3 distinct models. The game supports 2 to
4 total players, so an empty opponent list or more than 3 opponents raises a
setup error.

## Custom Agents

Custom agents can subclass `BaseAgent` or implement the same methods:

```python
from kbench_liarsbar import BaseAgent, ChallengeDecision, PlayDecision


class MyAgent(BaseAgent):
    def choose_play(self, context):
        return PlayDecision(
            played_cards=[context.legal_cards[0]],
            behavior="I place one card face down.",
            play_reason="I have a legal card available.",
        )

    def choose_challenge(self, context):
        return ChallengeDecision(
            was_challenged=False,
            challenge_reason="The previous claim is plausible.",
        )
```

Invalid custom-agent or LLM output raises an error instead of falling back to a
default action. This keeps behavior analysis faithful to the actual player.

## Local Development

```bash
python -m compileall -q src
python -m pytest
```

## Acknowledgements

This project reimplements Liars Bar game mechanics for Kaggle Benchmarks and was
informed by the public [LYiHub/liars-bar-llm](https://github.com/LYiHub/liars-bar-llm)
project structure and prompts.

This project is intended for research and benchmarking purposes only.
