#!/usr/bin/env python3
"""Evaluate trained StarCraft scenario PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from starcraft import EvaluationConfig, load_scenarios, evaluate_map_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved StarCraft scenario policies.")
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Aftershock).")
    parser.add_argument("--maps-root", default="starcraft-maps", help="Root directory containing sc1-map/ and sc1-scen/.")
    parser.add_argument("--models-dir", default="models/starcraft", help="Directory where checkpoints are stored.")
    parser.add_argument("--train-episodes", type=int, default=5000, help="Episodes used during training (locates checkpoints).")
    parser.add_argument("--scenario-count", type=int, default=10, help="Number of scenarios to evaluate (0 for all).")
    parser.add_argument("--rollouts", type=int, default=5, help="Evaluation rollouts per scenario.")
    parser.add_argument("--device", default="cpu", help="Device for PPO inference (cpu or cuda).")
    parser.add_argument(
        "--max-steps-scale",
        type=float,
        default=None,
        help="Override the horizon scale (defaults to the value saved during training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps_root = Path(args.maps_root)
    models_dir = Path(args.models_dir)

    scen_path = maps_root / "sc1-scen" / f"{args.map_id}.map.scen"
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scen_path}")

    scenarios = load_scenarios(scen_path, limit=None if args.scenario_count == 0 else args.scenario_count)
    if not scenarios:
        raise ValueError(f"No scenarios found in {scen_path}")

    config = EvaluationConfig(
        map_id=args.map_id,
        scenarios=scenarios,
        models_dir=models_dir,
        train_episodes=args.train_episodes,
        rollouts=args.rollouts,
        device=args.device,
        max_steps_scale=args.max_steps_scale,
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
