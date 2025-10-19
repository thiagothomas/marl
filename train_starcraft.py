#!/usr/bin/env python3
"""Train PPO policies for Moving AI StarCraft map scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from starcraft import TrainingConfig, load_scenarios, train_map_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policies for StarCraft map scenarios.")
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Aftershock).")
    parser.add_argument("--maps-root", default="starcraft-maps", help="Root directory containing sc1-map/ and sc1-scen/.")
    parser.add_argument("--models-dir", default="models/starcraft", help="Directory to store trained checkpoints.")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes per scenario.")
    parser.add_argument("--device", default="cpu", help="Device for PPO (cpu or cuda).")
    parser.add_argument("--scenario-count", type=int, default=10, help="Number of scenarios to train (0 for all).")
    parser.add_argument(
        "--scenario-index",
        type=int,
        help="Zero-based scenario line index to train (overrides --scenario-count).",
    )
    parser.add_argument(
        "--scenario-id",
        help="Specific scenario id to train (e.g., Aftershock_line_000).",
    )
    parser.add_argument(
        "--max-steps-scale",
        type=float,
        default=None,
        help="Override the automatically derived episode horizon scale (default uses scenario bucket & path length).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps_root = Path(args.maps_root)
    models_dir = Path(args.models_dir)

    scen_path = maps_root / "sc1-scen" / f"{args.map_id}.map.scen"
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scen_path}")

    if args.scenario_id:
        all_scenarios = load_scenarios(scen_path)
        scenarios = [s for s in all_scenarios if s.scenario_id == args.scenario_id]
        if not scenarios:
            available = ", ".join(s.scenario_id for s in all_scenarios[:10])
            raise ValueError(
                f"Scenario id '{args.scenario_id}' not found in {scen_path}. "
                f"Examples: {available}..."
            )
    elif args.scenario_index is not None:
        all_scenarios = load_scenarios(scen_path)
        if args.scenario_index < 0 or args.scenario_index >= len(all_scenarios):
            raise ValueError(
                f"Scenario index {args.scenario_index} out of range (0-{len(all_scenarios)-1})."
            )
        scenarios = [all_scenarios[args.scenario_index]]
    else:
        scenarios = load_scenarios(scen_path, limit=None if args.scenario_count == 0 else args.scenario_count)
    if not scenarios:
        raise ValueError(f"No scenarios found in {scen_path}")

    config = TrainingConfig(
        map_id=args.map_id,
        scenarios=scenarios,
        episodes=args.episodes,
        models_dir=models_dir,
        device=args.device,
        max_steps_scale=args.max_steps_scale,
    )

    metadata_paths = train_map_scenarios(config)

    checkpoints_root = models_dir / args.map_id / f"episodes_{args.episodes}"
    print("=" * 70)
    print(f"Trained {len(metadata_paths)} scenarios for map '{args.map_id}'.")
    print(f"Checkpoints stored under {checkpoints_root}.")
    print("=" * 70)
    for scenario, metadata_path in zip(scenarios, metadata_paths):
        print(f"- {scenario.scenario_id}")
        print(f"  metadata  : {metadata_path}")
        print(f"  model dir : {metadata_path.parent}")


if __name__ == "__main__":
    main()
