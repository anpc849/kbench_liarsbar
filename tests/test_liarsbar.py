from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kbench_liarsbar import (
    BaseAgent,
    ChallengeDecision,
    DefaultLLMAgent,
    GameConfig,
    InvalidAgentError,
    PlayDecision,
    build_benchmark_config,
    run_liarsbar_game,
)
from kbench_liarsbar.runner import LiarsBarGame


class FirstLegalAgent(BaseAgent):
    def __init__(self, *, challenge=False):
        self.challenge = challenge

    def choose_play(self, context):
        return PlayDecision(
            played_cards=[context.hand[0]],
            behavior="slides one card forward with a steady expression",
            play_reason="play the first available card",
        )

    def choose_challenge(self, context):
        return ChallengeDecision(
            was_challenged=self.challenge,
            challenge_reason="configured deterministic challenge policy",
        )


class BadPlayAgent(BaseAgent):
    def choose_play(self, context):
        return PlayDecision(
            played_cards=["NotACard"],
            behavior="tries an impossible move",
            play_reason="bad test action",
        )

    def choose_challenge(self, context):
        return ChallengeDecision(False, "not used")


class FixedPlayAgent(BaseAgent):
    def __init__(self, played_cards, *, challenge=False):
        self.played_cards = list(played_cards)
        self.challenge = challenge

    def choose_play(self, context):
        return PlayDecision(
            played_cards=list(self.played_cards),
            behavior="fixed test play",
            play_reason="deterministic test play",
        )

    def choose_challenge(self, context):
        return ChallengeDecision(
            was_challenged=self.challenge,
            challenge_reason="deterministic test challenge",
        )


class FakeLLM:
    def __init__(self, model):
        self.model = model

    def prompt(self, *args, **kwargs):
        raise AssertionError("not called in config tests")

    def respond(self, *args, **kwargs):
        raise AssertionError("not called in config tests")


class ReflectionLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.model = "reflection-test"

    def prompt(self, prompt, schema=str):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("no more fake responses")
        return self.responses.pop(0)

    def respond(self, *args, **kwargs):
        raise AssertionError("not used")


def make_config(*agents, seed=3, **updates):
    return GameConfig(
        player_configs=[
            {"name": f"Player {index}", "agent": agent}
            for index, agent in enumerate(agents, start=1)
        ],
        seed=seed,
        enable_reflection=False,
        **updates,
    )


def test_deck_composition():
    game = LiarsBarGame(
        game_config=make_config(FirstLegalAgent(), FirstLegalAgent())
    )

    deck = game.create_deck()

    assert len(deck) == 20
    assert deck.count("Q") == 6
    assert deck.count("K") == 6
    assert deck.count("A") == 6
    assert deck.count("Joker") == 2


def test_valid_play_uses_target_or_joker():
    game = LiarsBarGame(
        game_config=make_config(FirstLegalAgent(), FirstLegalAgent())
    )
    game.target_card = "Q"

    assert game.is_valid_play(["Q", "Joker"])
    assert not game.is_valid_play(["Q", "A"])


def test_revolver_chambers_controls_hit_probability():
    game = LiarsBarGame(
        game_config=make_config(
            FirstLegalAgent(),
            FirstLegalAgent(),
            revolver_chambers=1,
        )
    )
    shooter = game.players[0]

    game.perform_penalty(shooter)

    assert not shooter.alive
    assert shooter.total_shots_taken == 1
    assert game.public_history[-1]["bullet_hit"] is True


def test_revolver_chambers_must_be_positive():
    with pytest.raises(ValueError, match="revolver_chambers"):
        LiarsBarGame(
            game_config=make_config(
                FirstLegalAgent(),
                FirstLegalAgent(),
                revolver_chambers=0,
            )
        )


def test_deal_cards_gives_alive_players_five_cards_and_skips_dead_players():
    game = LiarsBarGame(
        game_config=make_config(FirstLegalAgent(), FirstLegalAgent(), FirstLegalAgent())
    )
    game.players[1].alive = False
    game.players[1].hand = []

    game.deal_cards()

    assert len(game.players[0].hand) == 5
    assert game.players[1].hand == []
    assert len(game.players[2].hand) == 5


def test_successful_challenge_penalizes_bluffer():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["A"]),
            FixedPlayAgent(["Q"], challenge=True),
            revolver_chambers=3,
        )
    )
    player, challenger = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["A"]
    challenger.hand = ["Q"]
    player.bullet_position = 2
    player.chamber_position = 0

    game.play_turn()

    challenge = next(event for event in game.public_history if event["type"] == "challenge")
    shot = next(event for event in game.public_history if event["type"] == "shot")
    assert challenge["challenge_success"] is True
    assert shot["shooter"] == player.name
    assert player.shots_taken == 1
    assert challenger.shots_taken == 0


def test_failed_challenge_penalizes_challenger():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=True),
            revolver_chambers=3,
        )
    )
    player, challenger = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q"]
    challenger.hand = ["K"]
    challenger.bullet_position = 2
    challenger.chamber_position = 0

    game.play_turn()

    challenge = next(event for event in game.public_history if event["type"] == "challenge")
    shot = next(event for event in game.public_history if event["type"] == "shot")
    assert challenge["challenge_success"] is False
    assert shot["shooter"] == challenger.name
    assert player.shots_taken == 0
    assert challenger.shots_taken == 1


def test_challenge_reason_is_private_not_public_history():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=True),
            revolver_chambers=3,
        )
    )
    player, challenger = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q"]
    challenger.hand = ["K"]
    challenger.bullet_position = 2
    challenger.chamber_position = 0

    game.play_turn()

    challenge = next(event for event in game.public_history if event["type"] == "challenge")
    decision = next(
        event
        for event in game.decision_log
        if event["phase"] == "challenge" and event["player"] == challenger.name
    )
    assert "reason" not in challenge
    assert "deterministic test challenge" in decision["decision"]["challenge_reason"]


def test_winner_is_set_when_penalty_eliminates_player():
    game = LiarsBarGame(
        game_config=make_config(
            FirstLegalAgent(),
            FirstLegalAgent(),
            revolver_chambers=2,
        )
    )
    loser, survivor = game.players
    loser.bullet_position = 0
    loser.chamber_position = 0

    game.perform_penalty(loser)

    assert not loser.alive
    assert survivor.alive
    assert game.game_over is True
    assert game.winner == survivor.name


def test_turn_moves_to_next_player_after_no_challenge():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=False),
        )
    )
    player, next_player = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q", "A"]
    next_player.hand = ["K"]

    game.play_turn()

    assert game.current_player_idx == 1
    assert player.hand == ["A"]
    assert game.public_history[-1]["type"] == "no_challenge"
    assert "reason" not in game.public_history[-1]


def test_no_challenge_reason_is_only_in_owner_private_prompt():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=False),
            revolver_chambers=4,
        )
    )
    player, next_player = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q", "A"]
    next_player.hand = ["K"]

    game.play_turn()

    next_player_context = game.build_play_context(next_player, player)
    next_player_prompt = next_player_context.to_text()
    player_context = game.build_play_context(player, next_player)
    player_prompt = player_context.to_text()

    assert "deterministic test challenge" in next_player_prompt
    assert "deterministic test play" not in next_player_prompt
    assert "deterministic test challenge" not in player_prompt
    assert "deterministic test play" in player_prompt
    assert "{'type': 'no_challenge'" not in next_player_prompt
    assert "Player 2 did not challenge Player 1" in next_player_prompt
    assert "has a 25% chance to eliminate you" in next_player_prompt


def test_surviving_shooter_starts_next_round_after_penalty():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=True),
            revolver_chambers=3,
        )
    )
    player, challenger = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q"]
    challenger.hand = ["K"]
    challenger.bullet_position = 2
    challenger.chamber_position = 0

    game.play_turn()

    assert challenger.alive
    assert challenger.shots_taken == 1
    assert game.round_id == 2
    assert game.current_player_idx == 1
    assert game.public_history[-1]["type"] == "round_start"
    assert game.public_history[-1]["starting_player"] == challenger.name


def test_dead_shooter_is_skipped_when_next_round_starts():
    game = LiarsBarGame(
        game_config=make_config(
            FixedPlayAgent(["Q"]),
            FixedPlayAgent(["K"], challenge=True),
            FirstLegalAgent(),
            revolver_chambers=2,
        )
    )
    player, challenger, third = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    player.hand = ["Q"]
    challenger.hand = ["K"]
    third.hand = ["A"]
    challenger.bullet_position = 0
    challenger.chamber_position = 0

    game.play_turn()

    assert not challenger.alive
    assert game.game_over is False
    assert game.round_id == 2
    assert game.current_player_idx == 2
    assert game.public_history[-1]["starting_player"] == third.name


def test_system_challenge_runs_when_all_other_alive_players_have_no_cards():
    game = LiarsBarGame(
        game_config=make_config(
            FirstLegalAgent(),
            FirstLegalAgent(),
            revolver_chambers=3,
        )
    )
    current, other = game.players
    game.round_id = 1
    game.target_card = "Q"
    game.current_player_idx = 0
    current.hand = ["A"]
    other.hand = []
    current.bullet_position = 2
    current.chamber_position = 0

    game.play_turn()

    assert [event["type"] for event in game.public_history[-4:]] == [
        "play",
        "challenge",
        "shot",
        "round_start",
    ]
    assert game.public_history[-3]["challenger"] == "System"
    assert game.public_history[-3]["challenge_success"] is True
    assert current.shots_taken == 1


def test_private_context_does_not_expose_unrevealed_cards():
    game = LiarsBarGame(
        game_config=make_config(FirstLegalAgent(), FirstLegalAgent())
    )
    p1, p2 = game.players
    game.round_id = 1
    game.target_card = "Q"
    p1.hand = ["A", "K"]
    p2.hand = ["Q", "Joker"]
    public_play = {
        "type": "play",
        "round_id": 1,
        "turn": 1,
        "player": p1.name,
        "target_card": "Q",
        "claimed_count": 1,
        "remaining_count": 1,
        "behavior": "claims confidence",
    }
    game.public_history = [public_play]

    context = game.build_challenge_context(
        challenger=p2,
        challenged=p1,
        previous_play=public_play,
    )

    assert "actual_cards" not in context.to_text()
    assert "claimed 1 Q card" in context.to_text()
    assert context.hand == ["Q", "Joker"]


def test_invalid_agent_action_is_rejected_without_fallback():
    config = make_config(BadPlayAgent(), FirstLegalAgent(challenge=True))
    game = LiarsBarGame(game_config=config)
    game.deal_cards()
    game.target_card = "Q"
    game.round_id = 1
    game.current_player_idx = 0

    with pytest.raises(InvalidAgentError):
        game.play_turn()

    assert not game.decision_log


def test_full_game_produces_winner_and_replay():
    result = run_liarsbar_game(
        game_config=make_config(
            FirstLegalAgent(challenge=True),
            FirstLegalAgent(challenge=True),
            seed=11,
            max_rounds=40,
            max_turns=120,
        )
    )

    assert result["winner"] in {"Player 1", "Player 2"} or result["timeout"]
    assert result["game_log"]["events"]
    assert result["public_history"]


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_full_game_smoke_for_supported_player_counts(player_count):
    agents = [FirstLegalAgent(challenge=True) for _ in range(player_count)]

    result = run_liarsbar_game(
        game_config=make_config(
            *agents,
            seed=20 + player_count,
            revolver_chambers=1,
            max_rounds=20,
            max_turns=80,
        )
    )

    assert result["timeout"] is False
    assert result["winner"] in {f"Player {index}" for index in range(1, player_count + 1)}
    assert sum(1 for player in result["players"] if player["alive"]) == 1
    assert result["public_history"][-1]["type"] == "round_start" or result["game_log"]["events"]


def test_benchmark_config_uses_task_defined_llm_opponents():
    evaluated = FakeLLM("google/gemini-3.5-flash")
    models = {
        "anthropic/claude-haiku-4-5@20251001": FakeLLM("claude"),
        "deepseek-ai/deepseek-v3.2": FakeLLM("deepseek"),
        "openai/gpt-5.4-mini-2026-03-17": FakeLLM("gpt"),
    }
    fake_kbench = type("FakeKBench", (), {"llms": models})()

    config = build_benchmark_config(
        fake_kbench,
        evaluated,
        opponent_model_ids=[
            "anthropic/claude-haiku-4-5@20251001",
            "deepseek-ai/deepseek-v3.2",
            "openai/gpt-5.4-mini-2026-03-17",
        ],
    )

    assert len(config.player_configs) == 4
    assert config.player_configs[0]["evaluated"] is True
    names = [p["name"] for p in config.player_configs]
    assert len(set(names)) == 4
    assert "Evaluated" not in names
    assert not any(name.startswith("Opponent") for name in names)


def test_benchmark_config_seed_is_optional_by_default():
    fake_kbench = type(
        "FakeKBench",
        (),
        {"llms": {"anthropic/claude-haiku-4-5@20251001": FakeLLM("claude")}},
    )()

    config = build_benchmark_config(
        fake_kbench,
        FakeLLM("eval"),
        opponent_model_ids=["anthropic/claude-haiku-4-5@20251001"],
    )

    assert config.seed is None


def test_benchmark_config_preserves_explicit_seed():
    fake_kbench = type(
        "FakeKBench",
        (),
        {"llms": {"anthropic/claude-haiku-4-5@20251001": FakeLLM("claude")}},
    )()

    config = build_benchmark_config(
        fake_kbench,
        FakeLLM("eval"),
        opponent_model_ids=["anthropic/claude-haiku-4-5@20251001"],
        seed=123,
    )

    assert config.seed == 123


def test_benchmark_config_allows_one_opponent_for_two_player_game():
    fake_kbench = type(
        "FakeKBench",
        (),
        {"llms": {"anthropic/claude-haiku-4-5@20251001": FakeLLM("claude")}},
    )()

    config = build_benchmark_config(
        fake_kbench,
        FakeLLM("eval"),
        opponent_model_ids=["anthropic/claude-haiku-4-5@20251001"],
    )

    assert len(config.player_configs) == 2
    assert config.player_configs[0]["evaluated"] is True
    assert config.evaluated_player_name == config.player_configs[0]["name"]


def test_benchmark_config_accepts_custom_player_names():
    fake_kbench = type(
        "FakeKBench",
        (),
        {"llms": {"anthropic/claude-haiku-4-5@20251001": FakeLLM("claude")}},
    )()

    config = build_benchmark_config(
        fake_kbench,
        FakeLLM("eval"),
        opponent_model_ids=["anthropic/claude-haiku-4-5@20251001"],
        player_names=["Mira", "Theo"],
    )

    assert [player["name"] for player in config.player_configs] == ["Mira", "Theo"]
    assert config.evaluated_player_name == "Mira"


def test_benchmark_config_rejects_zero_opponents():
    fake_kbench = type("FakeKBench", (), {"llms": {}})()

    with pytest.raises(ValueError):
        build_benchmark_config(
            fake_kbench,
            FakeLLM("eval"),
            opponent_model_ids=[],
        )


def test_benchmark_config_rejects_too_many_opponents():
    fake_kbench = type("FakeKBench", (), {"llms": {}})()

    with pytest.raises(ValueError):
        build_benchmark_config(
            fake_kbench,
            FakeLLM("eval"),
            opponent_model_ids=["a", "b", "c", "d"],
        )


def test_benchmark_config_rejects_missing_opponent_model():
    fake_kbench = type("FakeKBench", (), {"llms": {}})()

    with pytest.raises(RuntimeError, match="not available"):
        build_benchmark_config(
            fake_kbench,
            FakeLLM("eval"),
            opponent_model_ids=["a", "b", "c"],
        )


def test_default_llm_agent_uses_five_retries_by_default():
    agent = DefaultLLMAgent(ReflectionLLM([]), llm_pause_seconds=0)

    assert agent.max_retries == 5


def test_default_llm_agent_uses_nested_visible_chat(monkeypatch):
    calls = []
    state = {"active": False}

    class FakeChatContext:
        def __init__(self, name, orphan):
            self.name = name
            self.orphan = orphan

        def __enter__(self):
            calls.append((self.name, self.orphan))
            state["active"] = True

        def __exit__(self, exc_type, exc, tb):
            state["active"] = False

    class FakeChats:
        @staticmethod
        def new(name, orphan):
            return FakeChatContext(name, orphan)

    class ChatAwareLLM:
        def prompt(self, prompt, schema=str):
            assert state["active"] is True
            return "ok"

    monkeypatch.setitem(
        sys.modules,
        "kaggle_benchmarks",
        SimpleNamespace(chats=FakeChats),
    )
    agent = DefaultLLMAgent(ChatAwareLLM(), llm_pause_seconds=0)
    context = SimpleNamespace(player_name="Mira")

    result = agent._prompt_llm(
        context=context,
        phase="play",
        attempt=0,
        prompt="private prompt",
        schema=str,
    )

    assert result == "ok"
    assert calls == [("liarsbar-Mira-play-attempt-1", False)]


def test_reflection_retries_with_error_hint_and_parses_json():
    llm = ReflectionLLM(
        [
            "not json",
            '{"opinions": {"Opponent": "careful challenger after revealed bluffs"}}',
        ]
    )
    agent = DefaultLLMAgent(llm, llm_pause_seconds=0)
    context = type(
        "Context",
        (),
        {
            "player_name": "Self",
            "phase": "reflect",
            "round_id": 2,
            "alive_players": ["Self", "Opponent"],
            "to_text": lambda self: "public history only",
        },
    )()

    opinions = agent.reflect(context)

    assert opinions["Opponent"] == "careful challenger after revealed bluffs"
    assert len(llm.prompts) == 2
    assert "Validation error:" in llm.prompts[1]
    assert "not json" in llm.prompts[1]


def test_play_retry_prompt_includes_error_hint_and_retries():
    llm = ReflectionLLM(
        [
            {
                "played_cards": ["NotACard"],
                "behavior": "I make an invalid play.",
                "play_reason": "bad test response",
            },
            {
                "played_cards": ["Q"],
                "behavior": "I place one card forward.",
                "play_reason": "use a legal card after the correction",
            },
        ]
    )
    agent = DefaultLLMAgent(llm, llm_pause_seconds=0)
    context = type(
        "Context",
        (),
        {
            "player_name": "Self",
            "phase": "play",
            "round_id": 1,
            "target_card": "Q",
            "hand": ["Q", "K"],
            "legal_cards_text": lambda self: "Q, K",
            "legal_card_counts": [1, 2],
            "to_text": lambda self: "Player: Self\nYour hand: Q, K",
        },
    )()

    decision = agent.choose_play(context)

    assert decision.played_cards == ["Q"]
    assert len(llm.prompts) == 2
    assert "Validation error:" in llm.prompts[1]
    assert "Illegal cards" in llm.prompts[1]


def test_human_play_controls_hide_challenge_decision():
    from kbench_liarsbar.gradio_app import HumanRequest, visible_human_controls

    play_request = HumanRequest(
        run_id="test",
        player_name="Human",
        phase="play",
        round_id=1,
        target_card="A",
        hand=["A", "K"],
        legal_card_counts=[1, 2],
        prompt="Choose cards.",
        card_choices=["1: A", "2: K"],
    )
    challenge_request = HumanRequest(
        run_id="test",
        player_name="Human",
        phase="challenge",
        round_id=1,
        target_card="A",
        hand=["A", "K"],
        legal_card_counts=[],
        prompt="Decide whether to challenge.",
        card_choices=[],
    )

    play_updates = visible_human_controls(play_request)
    challenge_updates = visible_human_controls(challenge_request)

    play_challenge_update = play_updates[4]
    challenge_challenge_update = challenge_updates[4]
    assert play_challenge_update["visible"] is False
    assert play_challenge_update["choices"] == []
    assert play_challenge_update["interactive"] is False
    assert challenge_challenge_update["visible"] is True
    assert challenge_challenge_update["interactive"] is True


def test_empty_play_behavior_requires_explicit_opt_in():
    from kbench_liarsbar.agent.validation import validate_play_decision

    payload = {
        "played_cards": ["Q"],
        "behavior": "",
        "play_reason": "selected a legal card",
    }

    with pytest.raises(InvalidAgentError, match="behavior"):
        validate_play_decision(payload, ["Q"])

    decision = validate_play_decision(payload, ["Q"], allow_empty_behavior=True)

    assert decision.behavior == ""


def test_empty_challenge_reason_requires_explicit_opt_in():
    from kbench_liarsbar.agent.validation import validate_challenge_decision

    payload = {"was_challenged": False, "challenge_reason": ""}

    with pytest.raises(InvalidAgentError, match="challenge_reason"):
        validate_challenge_decision(payload)

    decision = validate_challenge_decision(payload, allow_empty_reason=True)

    assert decision.challenge_reason == ""
