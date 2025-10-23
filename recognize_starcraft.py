#!/usr/bin/env python3
"""Evaluate trained StarCraft scenario PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from starcraft import EvaluationConfig, load_scenarios, evaluate_map_scenarios

DEFAULT_MAPS_ROOT = Path("starcraft-maps")
DEFAULT_MODELS_DIR = Path("models/starcraft")
DEFAULT_DEVICE = "cpu"
DEFAULT_ROLLOUTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved StarCraft scenario policies.")
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Aftershock).")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Episodes used during training (locates checkpoints).",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Zero-based index of the scenario to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps_root = DEFAULT_MAPS_ROOT
    models_dir = DEFAULT_MODELS_DIR

    scen_path = maps_root / "sc1-scen" / f"{args.map_id}.map.scen"
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scen_path}")

    scenarios = load_scenarios(scen_path)
    if not scenarios:
        raise ValueError(f"No scenarios found in {scen_path}")

    if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
        raise ValueError(
            f"Scenario index {args.scenario_index} out of range (0-{len(scenarios)-1})."
        )
    scenario = scenarios[args.scenario_index]

    config = EvaluationConfig(
        map_id=args.map_id,
        scenarios=[scenario],
        models_dir=models_dir,
        train_episodes=args.episodes,
        rollouts=DEFAULT_ROLLOUTS,
        device=DEFAULT_DEVICE,
        max_steps_scale=None,
    )

    results = evaluate_map_scenarios(config)

    print("=" * 70)
    print(f"Evaluated {len(results)} scenarios for map '{args.map_id}'.")
    print("=" * 70)
    for item in results:
        print(f"- {item['scenario_id']}")
        print(f"  success rate : {item['success_rate']:.2%}")
        print(f"  mean reward  : {item['mean_reward']:.3f} ± {item['std_reward']:.3f}")


if __name__ == "__main__":
    main()
