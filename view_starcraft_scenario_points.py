#!/usr/bin/env python3
"""
Interactive viewer for Moving AI StarCraft scenarios.

Overlay start/goal markers from a .scen file on top of the hi-res PNG map.

Example:
    python view_starcraft_scenario_points.py \\
        --scen-file starcraft-maps/generated-scen/aftershock_line_2300_variants.scen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from starcraft.scenario_utils import Coordinate, ScenarioRecord, read_scenario_file


DEFAULT_MAPS_ROOT = Path("starcraft-maps")


class ScenarioOverlayCanvas(QWidget):
    """
    Canvas that renders the StarCraft map with scenario start/goal markers.
    """

    def __init__(
        self,
        pixmap: QPixmap,
        records: Sequence[ScenarioRecord],
        *,
        highlight_index: int = 0,
        show_all: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not records:
            raise ValueError("ScenarioOverlayCanvas requires at least one scenario record")

        self._pixmap = pixmap
        self._records = list(records)
        self._highlight_index = max(0, min(highlight_index, len(self._records) - 1))
        self._show_all = show_all
        self._grid_width = records[0].width
        self._grid_height = records[0].height

        self._zoom = 1.0
        self._min_zoom = 0.5
        self._max_zoom = 6.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_mouse_pos = QPointF()

    def set_highlight_index(self, index: int) -> None:
        self._highlight_index = max(0, min(index, len(self._records) - 1))
        self.update()

    def toggle_show_all(self) -> None:
        self._show_all = not self._show_all
        self.update()

    def change_zoom(self, delta: float) -> None:
        previous_zoom = self._zoom
        self._zoom = float(max(self._min_zoom, min(self._max_zoom, self._zoom + delta)))
        if self._zoom == previous_zoom:
            return
        scaling_factor = self._zoom / previous_zoom
        self._pan *= scaling_factor
        self.update()

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        angle = event.angleDelta().y()
        if angle == 0:
            return
        step = 0.15 if angle > 0 else -0.15
        self.change_zoom(step)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            current_pos = event.position()
            delta = current_pos - self._last_mouse_pos
            self._last_mouse_pos = current_pos
            self._pan += delta
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _to_canvas_transform(self):
        base_scale = min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )
        scale = base_scale * self._zoom
        scaled_width = self._pixmap.width() * scale
        scaled_height = self._pixmap.height() * scale
        scaled = self._pixmap.scaled(
            int(scaled_width),
            int(scaled_height),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_offset = (self.width() - scaled.width()) / 2.0 + self._pan.x()
        y_offset = (self.height() - scaled.height()) / 2.0 + self._pan.y()
        grid_to_pixmap_scale_x = self._pixmap.width() / self._grid_width
        grid_to_pixmap_scale_y = self._pixmap.height() / self._grid_height
        pixmap_to_canvas_scale_x = scaled.width() / self._pixmap.width()
        pixmap_to_canvas_scale_y = scaled.height() / self._pixmap.height()

        def to_canvas(pt: Coordinate) -> QPointF:
            return QPointF(
                pt[0] * grid_to_pixmap_scale_x * pixmap_to_canvas_scale_x + x_offset,
                pt[1] * grid_to_pixmap_scale_y * pixmap_to_canvas_scale_y + y_offset,
            )

        return scaled, x_offset, y_offset, to_canvas

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        scaled, x_offset, y_offset, to_canvas = self._to_canvas_transform()
        painter.drawPixmap(int(x_offset), int(y_offset), scaled)

        highlight = self._records[self._highlight_index]

        def draw_marker(
            record: ScenarioRecord,
            *,
            start_color: QColor,
            goal_color: QColor,
            radius: int,
            line_alpha: int,
        ) -> None:
            painter.setPen(QPen(goal_color, 2))
            painter.setBrush(QColor(goal_color.red(), goal_color.green(), goal_color.blue(), 140))
            painter.drawEllipse(to_canvas(record.goal), radius, radius)

            painter.setPen(QPen(start_color, 2))
            painter.setBrush(QColor(start_color.red(), start_color.green(), start_color.blue(), 140))
            painter.drawEllipse(to_canvas(record.start), radius, radius)

            painter.setPen(QPen(QColor(255, 255, 255, line_alpha), 1, Qt.PenStyle.DashLine))
            painter.drawLine(to_canvas(record.start), to_canvas(record.goal))

        if self._show_all:
            base_goal = QColor(90, 160, 255, 180)
            base_start = QColor(120, 255, 120, 180)
            for idx, record in enumerate(self._records):
                if idx == self._highlight_index:
                    continue
                draw_marker(
                    record,
                    start_color=base_start,
                    goal_color=base_goal,
                    radius=4,
                    line_alpha=100,
                )

        draw_marker(
            highlight,
            start_color=QColor(60, 230, 120),
            goal_color=QColor(70, 180, 255),
            radius=6,
            line_alpha=200,
        )


class ScenarioViewerWindow(QMainWindow):
    """
    Main window managing the overlay canvas and status text.
    """

    def __init__(
        self,
        pixmap: QPixmap,
        records: Sequence[ScenarioRecord],
        *,
        highlight_index: int,
        show_all: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        map_id = records[0].map_id
        self.setWindowTitle(f"Scenario Positions - {map_id}")

        self._records = list(records)
        self._highlight_index = max(0, min(highlight_index, len(self._records) - 1))

        self._canvas = ScenarioOverlayCanvas(
            pixmap,
            self._records,
            highlight_index=self._highlight_index,
            show_all=show_all,
        )

        self._status = QLabel(self)
        self._status.setStyleSheet("color: #f5f5f5; background-color: #202020; padding: 6px;")
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status)

        self.setCentralWidget(container)
        self._show_all = show_all
        self._update_status()

    def _update_status(self) -> None:
        record = self._records[self._highlight_index]
        self._status.setText(
            (
                f"Scenario {self._highlight_index}  |  bucket={record.bucket}  "
                f"|  start=({record.start[0]}, {record.start[1]}) → "
                f"goal=({record.goal[0]}, {record.goal[1]})  "
                f"|  optimal={record.optimal_length:.3f}  "
                "|  ←/→ to change  |  +/- to zoom  |  drag to pan  |  H toggle others  |  0 reset zoom"
            )
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self._highlight_index = (self._highlight_index + 1) % len(self._records)
            self._canvas.set_highlight_index(self._highlight_index)
            self._update_status()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self._highlight_index = (self._highlight_index - 1) % len(self._records)
            self._canvas.set_highlight_index(self._highlight_index)
            self._update_status()
            event.accept()
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._canvas.change_zoom(0.2)
            event.accept()
            return
        if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._canvas.change_zoom(-0.2)
            event.accept()
            return
        if key in (Qt.Key.Key_0,):
            self._canvas.reset_zoom()
            event.accept()
            return
        if key in (Qt.Key.Key_H,):
            self._canvas.toggle_show_all()
            self._show_all = not self._show_all
            self._update_status()
            event.accept()
            return
        super().keyPressEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display start/goal markers from a Moving AI scenario file."
    )
    parser.add_argument(
        "--scen-file",
        required=True,
        type=Path,
        help="Path to the .scen file to visualize (e.g. generated variants).",
    )
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=DEFAULT_MAPS_ROOT,
        help="Root directory containing sc1-png/<map>.png.",
    )
    parser.add_argument(
        "--highlight-index",
        type=int,
        default=0,
        help="Scenario index to highlight initially.",
    )
    parser.add_argument(
        "--hide-others",
        action="store_true",
        help="Hide non-highlighted markers (toggle inside viewer with H).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    header, records = read_scenario_file(args.scen_file)
    if not header.lower().startswith("version"):
        raise ValueError(f"Unexpected scenario header: {header}")

    if not records:
        raise ValueError(f"No scenarios found in {args.scen_file}")

    map_ids = {record.map_id for record in records}
    if len(map_ids) != 1:
        raise ValueError(
            f"Scenario file {args.scen_file} mixes multiple maps: {sorted(map_ids)}"
        )
    map_id = map_ids.pop()

    png_path = args.maps_root / "sc1-png" / f"{map_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"Map PNG not found: {png_path}")

    app = QApplication(sys.argv)

    pixmap = QPixmap(str(png_path))
    if pixmap.isNull():
        raise RuntimeError(f"Failed to load PNG map from {png_path}")
    window = ScenarioViewerWindow(
        pixmap,
        records,
        highlight_index=args.highlight_index,
        show_all=not args.hide_others,
    )
    window.resize(pixmap.width(), pixmap.height())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
