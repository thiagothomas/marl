#!/usr/bin/env python3
"""
Visualize trained StarCraft PPO policies on top of the Moving AI map PNG.

This viewer loads a saved checkpoint, replays the agent policy greedily,
and animates its trajectory over the hi-res map background.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from starcraft import (
    StarCraftScenarioEnv,
    load_grid,
    load_scenarios,
    derive_runtime_parameters,
    load_runtime_from_metadata,
    RuntimeParameters,
)
from ml.ppo import PPOAgent


Coordinate = Tuple[int, int]

DEFAULT_MAPS_ROOT = Path("starcraft-maps")
DEFAULT_MODELS_DIR = Path("models/starcraft")
DEFAULT_DEVICE = "cpu"
DEFAULT_STEP_INTERVAL_MS = 60
DEFAULT_DETERMINISTIC = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained StarCraft PPO policy on the map PNG."
    )
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Aftershock).")
    parser.add_argument(
        "--scen-file",
        type=Path,
        default=None,
        help=(
            "Optional override for the scenario file path. "
            "If omitted, defaults to starcraft-maps/sc1-scen/<map>.map.scen."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Episode count used during training (locates the checkpoint).",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Zero-based index into the scenario file.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=DEFAULT_DETERMINISTIC,
        help="Replay using greedy actions instead of sampling from the policy.",
    )
    parser.add_argument(
        "--step-interval-ms",
        type=int,
        default=DEFAULT_STEP_INTERVAL_MS,
        metavar="MS",
        help=(
            "Delay between visualization steps in milliseconds; "
            "smaller values speed up the animation."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Base directory containing trained model checkpoints.",
    )
    return parser.parse_args()


class AgentRunner:
    """Small helper that steps a trained PPO agent through the environment."""

    def __init__(
        self,
        agent: PPOAgent,
        env_factory: Callable[[], StarCraftScenarioEnv],
        *,
        deterministic: bool,
    ) -> None:
        self._env_factory = env_factory
        self._env = env_factory()
        self._agent = agent
        self._deterministic = deterministic

        self._obs, info = self._env.reset()
        start_position = info.get("position")
        if start_position is None:
            raise RuntimeError("Environment reset did not provide a starting position.")

        self.positions: List[Coordinate] = [tuple(start_position)]
        self.rewards: List[float] = []

        self.terminated = False
        self.truncated = False
        self.steps = 0
        self.last_reward = 0.0

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    def step(self) -> Coordinate:
        if self.terminated or self.truncated:
            return self.positions[-1]

        action_probs = self._agent.get_action_probabilities(self._obs)
        if self._deterministic:
            action = int(np.argmax(action_probs))
        else:
            probs = np.asarray(action_probs, dtype=np.float64)
            probs_sum = probs.sum()
            if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                action = int(np.argmax(action_probs))
            else:
                probs = probs / probs_sum
                action = int(np.random.choice(len(probs), p=probs))

        self._obs, reward, terminated, truncated, info = self._env.step(action)
        position = info.get("position")
        if position is None:
            raise RuntimeError("Environment step did not return a position in info.")

        self.positions.append(tuple(position))
        self.rewards.append(float(reward))
        self.last_reward = float(reward)
        self.steps += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        return self.positions[-1]


class MapCanvas(QWidget):
    """Widget that draws the map background and the agent trajectory."""

    def __init__(
        self,
        pixmap: QPixmap,
        start: Coordinate,
        goal: Coordinate,
        grid_width: int,
        grid_height: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self._start = start
        self._goal = goal
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._positions: Sequence[Coordinate] = [start]
        self._zoom = 1.0
        self._min_zoom = 0.5
        self._max_zoom = 6.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_mouse_pos = QPointF()

    def set_positions(self, positions: Sequence[Coordinate]) -> None:
        self._positions = positions
        self.update()

    def change_zoom(self, delta: float) -> None:
        previous_zoom = self._zoom
        self._zoom = float(np.clip(self._zoom + delta, self._min_zoom, self._max_zoom))
        if self._zoom == previous_zoom:
            return
        # Adjust pan so zoom centers on the same visual point.
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

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Scale background while keeping aspect ratio and centered.
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
        painter.drawPixmap(int(x_offset), int(y_offset), scaled)

        if not self._positions:
            return

        # Calculate scale from grid coordinates to pixmap pixels
        grid_to_pixmap_scale_x = self._pixmap.width() / self._grid_width
        grid_to_pixmap_scale_y = self._pixmap.height() / self._grid_height

        # Calculate scale from pixmap to canvas
        pixmap_to_canvas_scale_x = scaled.width() / self._pixmap.width()
        pixmap_to_canvas_scale_y = scaled.height() / self._pixmap.height()

        def to_canvas(pt: Coordinate) -> QPointF:
            # Convert grid coordinates to pixmap pixel coordinates, then to canvas
            return QPointF(
                pt[0] * grid_to_pixmap_scale_x * pixmap_to_canvas_scale_x + x_offset,
                pt[1] * grid_to_pixmap_scale_y * pixmap_to_canvas_scale_y + y_offset,
            )

        # Draw goal marker.
        painter.setPen(QPen(QColor(50, 150, 255), 3))
        painter.setBrush(QColor(50, 150, 255, 120))
        painter.drawEllipse(to_canvas(self._goal), 6, 6)

        # Draw start marker.
        painter.setPen(QPen(QColor(120, 255, 120), 3))
        painter.setBrush(QColor(120, 255, 120, 120))
        painter.drawEllipse(to_canvas(self._start), 6, 6)

        # Draw trajectory.
        painter.setPen(QPen(QColor(80, 160, 255), 2))
        last_point: QPointF | None = None
        for coordinate in self._positions:
            point = to_canvas(coordinate)
            if last_point is not None:
                painter.drawLine(last_point, point)
            last_point = point

        # Draw agent.
        painter.setPen(QPen(Qt.GlobalColor.red, 3))
        painter.setBrush(QColor(255, 80, 80, 180))
        painter.drawEllipse(to_canvas(self._positions[-1]), 7, 7)


class VisualizerWindow(QMainWindow):
    def __init__(
        self,
        map_pixmap: QPixmap,
        runner: AgentRunner,
        scenario_id: str,
        start: Coordinate,
        goal: Coordinate,
        grid_width: int,
        grid_height: int,
        runtime: RuntimeParameters,
        step_interval_ms: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"StarCraft Policy Viewer - {scenario_id}")
        self._runner = runner

        self._canvas = MapCanvas(
            map_pixmap,
            start=start,
            goal=goal,
            grid_width=grid_width,
            grid_height=grid_height
        )

        self._status = QLabel(self)
        self._status.setStyleSheet("color: #EEEEEE; background-color: #222222; padding: 6px;")
        run_mode = "deterministic" if runner.deterministic else "sampling"
        self._status_prefix = (
            f"Start ({start[0]}, {start[1]}) → goal ({goal[0]}, {goal[1]}) "
            f"| max_steps={runtime.max_steps} "
            f"| mode={run_mode} "
            "| +/- or wheel to zoom, drag to pan, 0 to reset."
        )

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status)
        self.setCentralWidget(container)

        self._canvas.set_positions(self._runner.positions)
        self._status.setText(self._status_prefix)

        self._timer = QTimer(self)
        self._timer.setInterval(step_interval_ms)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def _advance(self) -> None:
        if self._runner.terminated or self._runner.truncated:
            self._timer.stop()
            status = "Goal reached" if self._runner.terminated else "Stopped (max steps reached)"
            self._status.setText(f"{self._status_prefix}  {status} after {self._runner.steps} steps.")
            return

        position = self._runner.step()
        self._canvas.set_positions(self._runner.positions)

        self._status.setText(
            f"{self._status_prefix}  Step {self._runner.steps} | pos=({position[0]}, {position[1]}) "
            f"| reward={self._runner.last_reward:.3f}"
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
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
        super().keyPressEvent(event)


def main() -> None:
    args = parse_args()

    maps_root = DEFAULT_MAPS_ROOT
    scen_path = args.scen_file if args.scen_file is not None else maps_root / "sc1-scen" / f"{args.map_id}.map.scen"
    png_path = maps_root / "sc1-png" / f"{args.map_id}.png"

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
            f"Available map ids: {sorted(scenario_map_ids)}"
        )

    if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
        raise ValueError(
            f"Scenario index {args.scenario_index} out of range (0-{len(scenarios)-1})."
        )
    scenario = scenarios[args.scenario_index]
    if scenario.map_id != args.map_id:
        raise ValueError(
            f"Scenario at index {args.scenario_index} in {scen_path} is for map "
            f"'{scenario.map_id}', but --map-id requested '{args.map_id}'."
        )

    app = QApplication(sys.argv)

    pixmap = QPixmap(str(png_path))
    if pixmap.isNull():
        raise RuntimeError(f"Failed to load PNG map from {png_path}")

    models_dir = Path(args.models_dir) / args.map_id
    policy_dir = models_dir / f"episodes_{args.episodes}" / scenario.scenario_id / "PPOAgent"

    metadata_runtime = load_runtime_from_metadata(policy_dir / "config.json", scenario)
    if metadata_runtime is not None:
        runtime = metadata_runtime
    else:
        runtime = derive_runtime_parameters(
            scenario,
            None,
            allow_diagonal_actions=True,
            view_mode="moore",
            view_radius=1,
        )

    grid = load_grid(scenario.map_path)

    def env_factory(scenario=scenario, runtime=runtime):
        return StarCraftScenarioEnv(
            grid=grid,
            scenario=scenario,
            max_steps=runtime.max_steps,
            step_penalty=runtime.step_penalty,
            invalid_penalty=runtime.invalid_penalty,
            goal_reward=runtime.goal_reward,
            progress_scale=runtime.progress_scale,
            allow_diagonal_actions=runtime.allow_diagonal_actions,
            view_mode=runtime.view_mode,
            view_radius=runtime.view_radius,
        )

    agent = PPOAgent(
        env_name=env_factory,
        models_dir=str(models_dir),
        goal_hypothesis=scenario.scenario_id,
        episodes=args.episodes,
        device=DEFAULT_DEVICE,
    )

    try:
        agent.load_model()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Checkpoint not found for scenario {scenario.scenario_id}. "
            f"Expected under {policy_dir}/"
        ) from exc

    runner = AgentRunner(
        agent=agent,
        env_factory=env_factory,
        deterministic=args.deterministic,
    )

    window = VisualizerWindow(
        map_pixmap=pixmap,
        runner=runner,
        scenario_id=scenario.scenario_id,
        start=scenario.start,
        goal=scenario.goal,
        grid_width=grid.shape[1],
        grid_height=grid.shape[0],
        runtime=runtime,
        step_interval_ms=max(1, int(args.step_interval_ms)),
    )
    window.resize(720, 780)
    window.show()

    exit_code = app.exec()
    if exit_code != 0:
        QMessageBox.critical(
            None,
            "Visualizer Error",
            "Visualization ended with a non-zero exit code.",
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
