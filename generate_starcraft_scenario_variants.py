#!/usr/bin/env python3
"""
Utility for cloning StarCraft Moving AI scenarios with nearby start/goal pairs.

Example:
    python generate_starcraft_scenario_variants.py \\
        --map Aftershock \\
        --line-base 2300 \\
        --num-scenarios 3
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List

from starcraft.scenario_utils import (
    Coordinate,
    ScenarioRecord,
    read_scenario_file,
    write_scenario_file,
)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def jitter_coordinate(
    coord: Coordinate,
    rng: random.Random,
    max_offset: int,
    width: int,
    height: int,
) -> Coordinate:
    """Return a coordinate nudged by at most `max_offset` in each axis."""
    dx = rng.randint(-max_offset, max_offset)
    dy = rng.randint(-max_offset, max_offset)
    x = clamp(coord[0] + dx, 0, width - 1)
    y = clamp(coord[1] + dy, 0, height - 1)
    return x, y


def format_output_filename(map_id: str, base_line: int, output: Path | None) -> Path:
    if output is not None:
        return output

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "starcraft-maps" / "generated-scen"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{map_id.lower()}_line_{base_line}_variants.scen"


def resolve_scenario_path(map_arg: str, scen_dir: Path) -> Tuple[str, Path]:
    """
    Resolve a scenario file by map identifier, tolerating case differences.
    Returns the canonical map id (matching the file casing) and the path.
    """
    candidate = scen_dir / f"{map_arg}.map.scen"
    if candidate.exists():
        canonical = candidate.name[: -len(".map.scen")]
        return canonical, candidate

    normalized = map_arg.lower()
    for path in scen_dir.glob("*.map.scen"):
        canonical = path.name[: -len(".map.scen")]
        if canonical.lower() == normalized:
            return canonical, path

    raise FileNotFoundError(
        f"Could not find scenario file for map '{map_arg}' under {scen_dir}"
    )


def approximate_path_cost(
    base: ScenarioRecord, candidate_start: Coordinate, candidate_goal: Coordinate
) -> float:
    """
    Estimate the optimal path length for a nearby variation.
    Uses the base record's optimal length adjusted by the difference in straight-line
    distance. This keeps the value close to the original without running search.
    """
    base_dx = base.start[0] - base.goal[0]
    base_dy = base.start[1] - base.goal[1]
    base_distance = math.hypot(base_dx, base_dy)

    cand_dx = candidate_start[0] - candidate_goal[0]
    cand_dy = candidate_start[1] - candidate_goal[1]
    candidate_distance = math.hypot(cand_dx, cand_dy)

    adjustment = candidate_distance - base_distance
    estimate = max(base.optimal_length + adjustment, 0.0)
    return round(estimate, 8)


def generate_variants(
    base_record: ScenarioRecord,
    num_variants: int,
    rng: random.Random,
    max_offset: int,
) -> List[ScenarioRecord]:
    seen_pairs = {(base_record.start, base_record.goal)}
    variants: List[ScenarioRecord] = []

    attempts = 0
    while len(variants) < num_variants:
        attempts += 1
        if attempts > num_variants * 50:
            raise RuntimeError(
                "Unable to generate enough unique start/goal pairs within offset bounds"
            )

        start_variant = jitter_coordinate(
            base_record.start, rng, max_offset, base_record.width, base_record.height
        )
        goal_variant = jitter_coordinate(
            base_record.goal, rng, max_offset, base_record.width, base_record.height
        )

        if (start_variant, goal_variant) in seen_pairs:
            continue

        cost = approximate_path_cost(base_record, start_variant, goal_variant)
        max_adjustment = math.hypot(max_offset, max_offset)
        lower_bound = max(base_record.optimal_length - max_adjustment, 0.0)
        upper_bound = base_record.optimal_length + max_adjustment
        cost = round(clamp_float(cost, lower_bound, upper_bound), 8)
        variant = ScenarioRecord(
            bucket=base_record.bucket,
            map_name=base_record.map_name,
            width=base_record.width,
            height=base_record.height,
            start=start_variant,
            goal=goal_variant,
            optimal_length=cost,
        )

        seen_pairs.add((start_variant, goal_variant))
        variants.append(variant)

    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nearby StarCraft scenario variants."
    )
    parser.add_argument(
        "--map",
        required=True,
        help="Map identifier (e.g. Aftershock). Should match <Map>.map.scen file.",
    )
    parser.add_argument(
        "--line-base",
        type=int,
        required=True,
        help="Zero-based index of the scenario line to clone (excludes header).",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        required=True,
        help="Number of variant scenarios to create in addition to the base line.",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=2,
        help="Maximum coordinate offset (in tiles) applied to start/goal positions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible offsets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .scen path. Defaults to starcraft-maps/generated-scen/...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_scenarios <= 0:
        raise ValueError("--num-scenarios must be a positive integer")

    if args.max_offset < 0:
        raise ValueError("--max-offset must be non-negative")

    repo_root = Path(__file__).resolve().parent
    scen_dir = repo_root / "starcraft-maps" / "sc1-scen"
    map_id, scen_path = resolve_scenario_path(args.map, scen_dir)

    _, data_lines = read_scenario_file(scen_path)

    if args.line_base < 0 or args.line_base >= len(data_lines):
        raise IndexError(
            f"--line-base {args.line_base} out of range. File has {len(data_lines)} lines."
        )

    base_record = data_lines[args.line_base]

    rng = random.Random(args.seed)
    variants = generate_variants(base_record, args.num_scenarios, rng, args.max_offset)

    output_path = format_output_filename(map_id, args.line_base, args.output)
    write_scenario_file(
        output_path,
        [base_record, *variants],
        header="version 1",
    )

    print(
        f"Wrote {1 + len(variants)} scenarios to {output_path} "
        f"(base line {args.line_base}, map {map_id})"
    )


if __name__ == "__main__":
    main()
