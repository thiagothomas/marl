from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class ScenarioRecord:
    """
    Lightweight representation of a Moving AI scenario line.
    """

    bucket: int
    map_name: str
    width: int
    height: int
    start: Coordinate
    goal: Coordinate
    optimal_length: float

    @property
    def map_id(self) -> str:
        """
        Return the map identifier without the `.map` suffix.
        """
        return Path(self.map_name).stem

    @classmethod
    def from_line(cls, raw: str) -> "ScenarioRecord":
        fields = raw.strip().split()
        if len(fields) != 9:
            raise ValueError(
                f"Expected 9 columns per scenario line, found {len(fields)} in: {raw}"
            )

        bucket = int(fields[0])
        map_name = fields[1]
        width = int(fields[2])
        height = int(fields[3])
        start = (int(fields[4]), int(fields[5]))
        goal = (int(fields[6]), int(fields[7]))
        optimal_length = float(fields[8])
        return cls(
            bucket=bucket,
            map_name=map_name,
            width=width,
            height=height,
            start=start,
            goal=goal,
            optimal_length=optimal_length,
        )

    def to_line(self) -> str:
        return (
            f"{self.bucket}\t{self.map_name}\t{self.width}\t{self.height}\t"
            f"{self.start[0]}\t{self.start[1]}\t{self.goal[0]}\t{self.goal[1]}\t"
            f"{self.optimal_length:.8f}"
        )


def parse_scenario_lines(lines: Iterable[str]) -> List[ScenarioRecord]:
    """
    Parse an iterable of scenario lines (excluding the header).
    """
    records: List[ScenarioRecord] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        records.append(ScenarioRecord.from_line(stripped))
    return records


def read_scenario_file(path: Path) -> Tuple[str, List[ScenarioRecord]]:
    """
    Read a Moving AI scenario file and return its header plus parsed records.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario file {path} does not exist")

    with path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\n") for line in handle if line.strip()]

    if not raw_lines:
        raise ValueError(f"Scenario file {path} is empty")

    header = raw_lines[0]
    if not header.lower().startswith("version"):
        raise ValueError(
            f"Scenario file {path} missing expected 'version' header (found: {header})"
        )
    records = parse_scenario_lines(raw_lines[1:])
    return header, records


def write_scenario_file(
    path: Path,
    records: Sequence[ScenarioRecord],
    header: str = "version 1",
) -> None:
    """
    Serialize scenario records to disk in Moving AI format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        for record in records:
            handle.write(record.to_line() + "\n")
