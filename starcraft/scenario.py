from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    map_id: str
    map_path: Path
    bucket: int
    width: int
    height: int
    start: Coordinate
    goal: Coordinate
    optimal_length: float
    index: int

    @property
    def scenario_id(self) -> str:
        return f"{self.map_id}_line_{self.index:03d}"


def parse_scenario_line(
    map_id: str,
    map_path: Path,
    scenario_file: Path,
    raw: str,
    line_index: int,
) -> Scenario:
    fields = raw.strip().split()
    if len(fields) != 9:
        raise ValueError(
            f"Scenario line {line_index} in {scenario_file} "
            f"expected 9 fields, found {len(fields)}"
        )

    bucket = int(fields[0])
    map_field = fields[1]
    width = int(fields[2])
    height = int(fields[3])
    start = (int(fields[4]), int(fields[5]))
    goal = (int(fields[6]), int(fields[7]))
    optimal = float(fields[8])

    if map_field != map_path.name:
        raise ValueError(
            f"Scenario line {line_index} in {scenario_file} refers to map '{map_field}' "
            f"but expected '{map_path.name}'"
        )

    return Scenario(
        map_id=map_id,
        map_path=map_path,
        bucket=bucket,
        width=width,
        height=height,
        start=start,
        goal=goal,
        optimal_length=optimal,
        index=line_index,
    )


def load_scenarios(scen_path: Path, limit: Optional[int] = None) -> List[Scenario]:
    """
    Load Moving AI scenario file (.scen) and return parsed Scenario objects.
    """
    if scen_path.suffix != ".scen":
        raise ValueError(f"Scenario path must end with .scen (got {scen_path})")

    map_filename = scen_path.stem
    map_id = Path(map_filename).stem  # removes potential .map
    map_path = scen_path.parent.parent / "sc1-map" / f"{map_id}.map"
    if not map_path.exists():
        raise FileNotFoundError(f"Map file {map_path} referenced by {scen_path} not found")

    with scen_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Scenario file {scen_path} is empty")

    if not lines[0].startswith("version"):
        raise ValueError(f"Scenario file {scen_path} missing version header")

    scenarios: List[Scenario] = []
    for idx, raw in enumerate(lines[1:]):
        scenario = parse_scenario_line(map_id, map_path, scen_path, raw, idx)
        scenarios.append(scenario)
        if limit is not None and len(scenarios) >= limit:
            break

    return scenarios
