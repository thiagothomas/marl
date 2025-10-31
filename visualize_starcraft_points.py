#!/usr/bin/env python3
"""Display all scenario start/goal points overlaid on the StarCraft map image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QWidget

from starcraft import Scenario, load_scenarios


Coordinate = Tuple[int, int]

DEFAULT_MAPS_ROOT = Path("starcraft-maps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay every scenario start and goal on the StarCraft map."
    )
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Archipelago).")
    parser.add_argument(
        "--scen-file",
        type=Path,
        default=None,
        help=(
            "Optional override for the scenario file path. "
            "Defaults to starcraft-maps/sc1-scen/<map-id>.map.scen."
        ),
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=4.0,
        help="Marker diameter (in pixels) used to draw start/goal points.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Alpha transparency for the markers (0.0 transparent, 1.0 opaque).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ScenarioPoints:
    starts: Sequence[Coordinate]
    goals: Sequence[Coordinate]


class ScenarioPointsWidget(QWidget):
    """Widget that draws the map and overlays all scenario start/goal points."""

    def __init__(
        self,
        pixmap: QPixmap,
        points: ScenarioPoints,
        *,
        marker_size: float,
        alpha: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self._points = points
        self._marker_size = max(1.0, marker_size)
        self._alpha = float(max(0.0, min(1.0, alpha)))
        self.setFixedSize(pixmap.size())

    @property
    def marker_size(self) -> float:
        return self._marker_size

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawPixmap(0, 0, self._pixmap)

        start_pen = QPen(QColor(64, 147, 255))
        start_pen.setWidthF(self._marker_size)
        start_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        start_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        goal_pen = QPen(QColor(255, 92, 128))
        goal_pen.setWidthF(self._marker_size)
        goal_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        goal_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        start_color = QColor(start_pen.color())
        goal_color = QColor(goal_pen.color())
        start_color.setAlphaF(self._alpha)
        goal_color.setAlphaF(self._alpha)
        start_pen.setColor(start_color)
        goal_pen.setColor(goal_color)

        painter.setPen(start_pen)
        for point in self._points.starts:
            painter.drawPoint(QPointF(point[0], point[1]))

        painter.setPen(goal_pen)
        for point in self._points.goals:
            painter.drawPoint(QPointF(point[0], point[1]))

        painter.end()


class ScenarioPointsWindow(QMainWindow):
    """Main window that hosts the scenario points canvas and legend information."""

    def __init__(
        self,
        *,
        map_id: str,
        scenario_count: int,
        canvas: ScenarioPointsWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{map_id} scenario points")
        self.setCentralWidget(canvas)
        self._status_bar = QStatusBar(self)
        self._status_bar.setStyleSheet("color: #EEEEEE; background-color: #222222; padding: 4px;")
        info = QLabel(
            f"{scenario_count} scenarios  |  blue=start  |  pink=goal  |  marker={canvas.marker_size:.1f}px"
        )
        info.setStyleSheet("color: #EEEEEE;")
        self._status_bar.addPermanentWidget(info)
        self.setStatusBar(self._status_bar)


def collect_points(scenarios: Iterable[Scenario]) -> ScenarioPoints:
    starts: list[Coordinate] = []
    goals: list[Coordinate] = []
    for scenario in scenarios:
        starts.append(tuple(map(int, scenario.start)))
        goals.append(tuple(map(int, scenario.goal)))
    return ScenarioPoints(starts=starts, goals=goals)


def main() -> None:
    args = parse_args()

    scen_path = (
        args.scen_file
        if args.scen_file is not None
        else DEFAULT_MAPS_ROOT / "sc1-scen" / f"{args.map_id}.map.scen"
    )
    png_path = DEFAULT_MAPS_ROOT / "sc1-png" / f"{args.map_id}.png"

    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scen_path}")
    if not png_path.exists():
        raise FileNotFoundError(f"Map PNG not found: {png_path}")

    scenarios = load_scenarios(scen_path)
    if not scenarios:
        raise ValueError(f"No scenarios found in {scen_path}")

    scenario_map_ids = {scenario.map_id for scenario in scenarios}
    if args.map_id not in scenario_map_ids:
        raise ValueError(
            f"Scenario file {scen_path} does not contain map '{args.map_id}'. "
            f"Found: {sorted(scenario_map_ids)}"
        )

    app = QApplication(sys.argv)

    pixmap = QPixmap(str(png_path))
    if pixmap.isNull():
        raise RuntimeError(f"Failed to load PNG map from {png_path}")

    points = collect_points(scenarios)
    canvas = ScenarioPointsWidget(
        pixmap=pixmap,
        points=points,
        marker_size=float(args.marker_size),
        alpha=float(args.alpha),
    )

    window = ScenarioPointsWindow(
        map_id=args.map_id,
        scenario_count=len(points.starts),
        canvas=canvas,
    )
    window.show()
    exit_code = app.exec()
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
