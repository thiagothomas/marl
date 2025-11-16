#!/usr/bin/env python3
"""Visualize StarCraft team PPO policies on the Moving AI map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
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
    RuntimeParameters,
    Scenario,
    StarCraftTeamEnv,
    TeamFormation,
    load_grid,
    load_team_formation,
)
from ml.ppo import PPOAgent


Coordinate = Tuple[int, int]

DEFAULT_TEAM_MODELS_DIR = Path("models/starcraft/team_policies")
DEFAULT_FORMATIONS_DIR = Path("models/starcraft/teams")
DEFAULT_DEVICE = "cpu"
DEFAULT_STEP_INTERVAL_MS = 60
DEFAULT_DETERMINISTIC = False

AGENT_COLORS = [
    QColor("#ff6b6b"),
    QColor("#4dabf7"),
    QColor("#ffd43b"),
    QColor("#20c997"),
    QColor("#845ef7"),
    QColor("#ffa94d"),
    QColor("#e64980"),
    QColor("#63e6be"),
    QColor("#94d82d"),
    QColor("#74c0fc"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a StarCraft team PPO policy on the map PNG."
    )
    parser.add_argument("--team-name", required=True, help="Team directory name to load.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Episode count used during team training.",
    )
    parser.add_argument(
        "--teams-dir",
        type=Path,
        default=DEFAULT_TEAM_MODELS_DIR,
        help="Base directory containing team policy checkpoints.",
    )
    parser.add_argument(
        "--map-id",
        type=str,
        default=None,
        help="Optional map identifier to disambiguate when multiple exist.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=DEFAULT_DETERMINISTIC,
        help="Replay using greedy joint actions instead of sampling.",
    )
    parser.add_argument(
        "--step-interval-ms",
        type=int,
        default=DEFAULT_STEP_INTERVAL_MS,
        metavar="MS",
        help="Delay between visualization steps in milliseconds.",
    )
    parser.add_argument(
        "--formations-dir",
        type=Path,
        default=DEFAULT_FORMATIONS_DIR,
        help="Directory that stores the single-agent formations (models/starcraft/teams).",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["sum", "mean"],
        default="sum",
        help="Reward aggregation used during training (only used if metadata is missing).",
    )
    return parser.parse_args()


class TeamRunner:
    """Step a trained joint policy and retain per-agent trajectories."""

    def __init__(
        self,
        agent: PPOAgent,
        env_factory: Callable[[], StarCraftTeamEnv],
        *,
        deterministic: bool,
        starts: Sequence[Coordinate],
        scenario_ids: Sequence[str],
    ) -> None:
        self._env_factory = env_factory
        self._env = env_factory()
        self._agent = agent
        self._deterministic = deterministic

        self._obs, _ = self._env.reset()
        self.positions: List[List[Coordinate]] = [
            [tuple(start)] for start in starts
        ]
        self.scenario_ids = list(scenario_ids)
        self.agent_success = [False] * len(starts)
        self.agent_done = [False] * len(starts)

        self.steps = 0
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def num_agents(self) -> int:
        return len(self.positions)

    def step(self) -> None:
        if self.terminated or self.truncated:
            return

        action = self._agent.act(self._obs, deterministic=self._deterministic)
        self._obs, reward, terminated, truncated, info = self._env.step(action)
        agent_infos = info.get("agent_info") if isinstance(info, dict) else None
        if not isinstance(agent_infos, list):
            agent_infos = [{}] * self.num_agents

        for idx in range(self.num_agents):
            payload = agent_infos[idx] if idx < len(agent_infos) else {}
            position = payload.get("position")
            if position is None:
                position = self.positions[idx][-1]
            self.positions[idx].append(tuple(position))

            terminated_flag = bool(payload.get("terminated"))
            truncated_flag = bool(payload.get("truncated")) and not terminated_flag
            if terminated_flag:
                self.agent_success[idx] = True
                self.agent_done[idx] = True
            elif truncated_flag:
                self.agent_done[idx] = True

        self.last_reward = float(reward)
        self.steps += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        if self.truncated and not self.terminated:
            for idx in range(self.num_agents):
                if not self.agent_done[idx]:
                    self.agent_done[idx] = True

    def status_lines(self) -> List[str]:
        lines: List[str] = []
        for idx, scenario_id in enumerate(self.scenario_ids):
            if self.agent_success[idx]:
                state = "✓ goal"
            elif self.agent_done[idx]:
                state = "stopped"
            else:
                state = "running"
            lines.append(f"{scenario_id}: {state}")
        return lines


class TeamMapCanvas(QWidget):
    """Canvas that draws the map background and multiple agent trajectories."""

    def __init__(
        self,
        pixmap: QPixmap,
        starts: Sequence[Coordinate],
        goals: Sequence[Coordinate],
        grid_width: int,
        grid_height: int,
        colors: Sequence[QColor],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self._starts = list(starts)
        self._goals = list(goals)
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._paths: Sequence[Sequence[Coordinate]] = [[start] for start in starts]
        self._colors = colors
        self._zoom = 1.0
        self._min_zoom = 0.5
        self._max_zoom = 6.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_mouse_pos = QPointF()

    def set_paths(self, paths: Sequence[Sequence[Coordinate]]) -> None:
        self._paths = paths
        self.update()

    def change_zoom(self, delta: float) -> None:
        previous_zoom = self._zoom
        self._zoom = float(np.clip(self._zoom + delta, self._min_zoom, self._max_zoom))
        if self._zoom == previous_zoom:
            return
        scaling_factor = self._zoom / previous_zoom
        self._pan *= scaling_factor
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

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

        if not self._paths:
            return

        grid_to_pixmap_scale_x = self._pixmap.width() / self._grid_width
        grid_to_pixmap_scale_y = self._pixmap.height() / self._grid_height
        pixmap_to_canvas_scale_x = scaled.width() / self._pixmap.width()
        pixmap_to_canvas_scale_y = scaled.height() / self._pixmap.height()

        def to_canvas(pt: Coordinate) -> QPointF:
            return QPointF(
                pt[0] * grid_to_pixmap_scale_x * pixmap_to_canvas_scale_x + x_offset,
                pt[1] * grid_to_pixmap_scale_y * pixmap_to_canvas_scale_y + y_offset,
            )

        for idx, goal in enumerate(self._goals):
            color = self._colors[idx % len(self._colors)]
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 130))
            painter.drawEllipse(to_canvas(goal), 6, 6)

        for idx, start in enumerate(self._starts):
            color = self._colors[idx % len(self._colors)]
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(to_canvas(start), 6, 6)

        for idx, path in enumerate(self._paths):
            if not path:
                continue
            color = self._colors[idx % len(self._colors)]
            painter.setPen(QPen(color, 2))
            last_point: QPointF | None = None
            for coordinate in path:
                point = to_canvas(coordinate)
                if last_point is not None:
                    painter.drawLine(last_point, point)
                last_point = point
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 200))
            painter.drawEllipse(last_point, 7, 7)

    # Interaction helpers -------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.angleDelta().y() > 0:
            self.change_zoom(0.1)
        else:
            self.change_zoom(-0.1)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = QPointF(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            delta = QPointF(event.position()) - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = QPointF(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Plus:
            self.change_zoom(0.1)
        elif event.key() == Qt.Key.Key_Minus:
            self.change_zoom(-0.1)
        elif event.key() == Qt.Key.Key_0:
            self.reset_view()


class TeamVisualizerWindow(QMainWindow):
    """Top-level window that displays the map canvas and status text."""

    def __init__(
        self,
        *,
        map_pixmap: QPixmap,
        runner: TeamRunner,
        starts: Sequence[Coordinate],
        goals: Sequence[Coordinate],
        scenario_ids: Sequence[str],
        grid_width: int,
        grid_height: int,
        runtime: RuntimeParameters,
        step_interval_ms: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("StarCraft Team Policy Viewer")
        self._runner = runner
        self._scenario_ids = list(scenario_ids)

        colors = AGENT_COLORS
        self._canvas = TeamMapCanvas(
            map_pixmap,
            starts=starts,
            goals=goals,
            grid_width=grid_width,
            grid_height=grid_height,
            colors=colors,
        )

        self._status = QLabel(self)
        self._status.setStyleSheet("color: #EEEEEE; background-color: #222222; padding: 6px;")
        run_mode = "deterministic" if runner.deterministic else "sampling"
        self._status_prefix = (
            f"max_steps={runtime.max_steps} | mode={run_mode} "
            "| +/- or wheel to zoom, drag to pan, 0 to reset."
        )

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status)
        self.setCentralWidget(container)

        self._timer = QTimer(self)
        self._timer.setInterval(step_interval_ms)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def _advance(self) -> None:
        if self._runner.terminated or self._runner.truncated:
            self._timer.stop()
            status = (
                "Team success"
                if self._runner.terminated
                else "Stopped (max steps or dead-end)"
            )
            self._canvas.set_paths(self._runner.positions)
            lines = " | ".join(self._runner.status_lines())
            self._status.setText(
                f"{self._status_prefix}  {status} after {self._runner.steps} steps.  {lines}"
            )
            return

        self._runner.step()
        self._canvas.set_paths(self._runner.positions)
        lines = " | ".join(self._runner.status_lines())
        self._status.setText(
            f"{self._status_prefix}  step={self._runner.steps} reward={self._runner.last_reward:.3f}  {lines}"
        )


# Helpers -----------------------------------------------------------------
def _extract_map_id_from_dir(policy_parent: Path, team_name: str) -> str | None:
    suffix = f"_team_{team_name}"
    name = policy_parent.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def _resolve_policy_dir(
    base_dir: Path,
    team_name: str,
    episodes: int,
    map_id_hint: str | None,
) -> Path:
    team_root = base_dir / team_name / f"episodes_{episodes}"
    if not team_root.exists():
        raise FileNotFoundError(f"Team episodes directory not found: {team_root}")

    candidates: List[Tuple[Path, str | None]] = []
    for entry in sorted(team_root.iterdir()):
        if not entry.is_dir():
            continue
        policy_dir = entry / "PPOAgent"
        if not policy_dir.exists():
            continue
        metadata_path = policy_dir / "config.json"
        map_id = None
        if metadata_path.exists():
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                map_id = raw.get("map_id")
            except json.JSONDecodeError:
                map_id = None
        if map_id is None:
            map_id = _extract_map_id_from_dir(policy_dir.parent, team_name)
        candidates.append((policy_dir, map_id))

    if not candidates:
        raise FileNotFoundError(f"No PPOAgent checkpoints found under {team_root}")

    if map_id_hint:
        for candidate, map_id in candidates:
            if map_id == map_id_hint:
                return candidate
        raise ValueError(
            f"Map '{map_id_hint}' not found for team '{team_name}' at episodes {episodes}."
        )

    unique_map_ids = {map_id for _, map_id in candidates if map_id}
    if len(candidates) > 1 and len(unique_map_ids) > 1:
        display = ", ".join(sorted(map_id for map_id in unique_map_ids if map_id))
        raise ValueError(
            "Multiple map checkpoints detected; provide --map-id to disambiguate. "
            f"Available: {display or 'unknown maps'}"
        )

    return candidates[0][0]


def _parse_scenario_index(scenario_id: str) -> int:
    if "_line_" in scenario_id:
        suffix = scenario_id.rsplit("_line_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    digits = "".join(ch for ch in scenario_id if ch.isdigit())
    return int(digits) if digits else 0


def _load_metadata(
    policy_dir: Path,
    formation_dir: Path,
    team_name: str,
    reward_mode_hint: str,
    episodes: int,
) -> Tuple[dict, Path]:
    metadata_path = policy_dir / "config.json"
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        return raw, metadata_path

    formation_path = formation_dir / team_name
    if not formation_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path} and formation directory missing: {formation_path}"
        )
    formation = load_team_formation(formation_path)
    synthetic = _synthesize_metadata_from_formation(
        formation,
        reward_mode_hint,
        episodes,
    )
    metadata_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    return synthetic, metadata_path


def _build_scenarios_and_runtimes(
    raw: dict,
    map_path: Path,
    grid_width: int,
    grid_height: int,
) -> Tuple[List[Scenario], List[RuntimeParameters], List[Coordinate], List[Coordinate], List[str]]:
    scenarios: List[Scenario] = []
    runtimes: List[RuntimeParameters] = []
    starts: List[Coordinate] = []
    goals: List[Coordinate] = []
    scenario_ids: List[str] = []

    agents = raw.get("agents") or []
    if not agents:
        raise ValueError("Metadata does not describe any agents.")

    for agent in agents:
        scenario_id = str(agent.get("scenario_id") or "unknown")
        start_raw = agent.get("start") or {}
        goal_raw = agent.get("goal") or {}
        start = (int(start_raw["x"]), int(start_raw["y"]))
        goal = (int(goal_raw["x"]), int(goal_raw["y"]))
        bucket = int(agent.get("bucket", 0))
        optimal = float(agent.get("optimal_length", 0.0))
        runtime_raw = agent.get("runtime") or {}

        scenario = Scenario(
            map_id=str(raw.get("map_id")),
            map_path=map_path,
            bucket=bucket,
            width=grid_width,
            height=grid_height,
            start=start,
            goal=goal,
            optimal_length=optimal,
            index=_parse_scenario_index(scenario_id),
        )
        scenarios.append(scenario)
        starts.append(start)
        goals.append(goal)
        scenario_ids.append(scenario_id)

        runtime = RuntimeParameters(
            max_steps=int(runtime_raw.get("max_steps", raw.get("aggregate_runtime", {}).get("max_steps", 1))),
            step_penalty=float(runtime_raw.get("step_penalty", 0.01)),
            invalid_penalty=float(runtime_raw.get("invalid_penalty", 0.04)),
            goal_reward=float(runtime_raw.get("goal_reward", 1.0)),
            scale_used=float(runtime_raw.get("scale_used", 1.0)),
            progress_scale=float(runtime_raw.get("progress_scale", 0.1)),
            allow_diagonal_actions=bool(runtime_raw.get("allow_diagonal_actions", True)),
            view_mode=str(runtime_raw.get("view_mode", "moore")),
            view_radius=int(runtime_raw.get("view_radius", 1)),
        )
        runtimes.append(runtime)

    return scenarios, runtimes, starts, goals, scenario_ids


def _aggregate_runtime(runtimes: Sequence[RuntimeParameters]) -> RuntimeParameters:
    if not runtimes:
        raise ValueError("At least one runtime is required to aggregate")
    base = runtimes[0]
    max_steps = max(runtime.max_steps for runtime in runtimes)
    return RuntimeParameters(
        max_steps=max_steps,
        step_penalty=base.step_penalty,
        invalid_penalty=base.invalid_penalty,
        goal_reward=base.goal_reward,
        scale_used=base.scale_used,
        progress_scale=base.progress_scale,
        allow_diagonal_actions=base.allow_diagonal_actions,
        view_mode=base.view_mode,
        view_radius=base.view_radius,
    )


def _runtime_to_dict(runtime: RuntimeParameters) -> dict:
    return {
        "max_steps": runtime.max_steps,
        "step_penalty": runtime.step_penalty,
        "invalid_penalty": runtime.invalid_penalty,
        "goal_reward": runtime.goal_reward,
        "scale_used": runtime.scale_used,
        "progress_scale": runtime.progress_scale,
        "allow_diagonal_actions": runtime.allow_diagonal_actions,
        "view_mode": runtime.view_mode,
        "view_radius": runtime.view_radius,
    }


def _synthesize_metadata_from_formation(
    formation: TeamFormation,
    reward_mode: str,
    episodes: int,
) -> dict:
    runtimes = [agent.runtime for agent in formation.agents]
    aggregate = _aggregate_runtime(runtimes)
    metadata = {
        "team_name": formation.team_name,
        "map_id": formation.map_id,
        "map_path": str(formation.map_path),
        "episodes": episodes,
        "num_agents": len(formation.agents),
        "reward_mode": reward_mode,
        "scenario_ids": formation.scenario_ids,
        "agents": [
            {
                "scenario_id": agent.scenario.scenario_id,
                "start": {"x": agent.scenario.start[0], "y": agent.scenario.start[1]},
                "goal": {"x": agent.scenario.goal[0], "y": agent.scenario.goal[1]},
                "bucket": agent.scenario.bucket,
                "optimal_length": agent.scenario.optimal_length,
                "runtime": _runtime_to_dict(agent.runtime),
            }
            for agent in formation.agents
        ],
        "aggregate_runtime": _runtime_to_dict(aggregate),
        "training": {
            "note": "synthetic metadata (training stats unavailable until checkpoint completes)",
        },
    }
    return metadata


def main() -> None:
    args = parse_args()
    teams_dir = Path(args.teams_dir)
    formations_dir = Path(args.formations_dir)

    policy_dir = _resolve_policy_dir(teams_dir, args.team_name, args.episodes, args.map_id)
    metadata, metadata_path = _load_metadata(
        policy_dir,
        formations_dir,
        args.team_name,
        args.reward_mode,
        args.episodes,
    )
    if "map_path" not in metadata or "map_id" not in metadata:
        raise ValueError(f"Metadata missing map information in {metadata_path}")
    map_path = Path(metadata["map_path"])
    if not map_path.is_absolute():
        map_path = (Path.cwd() / map_path).resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    grid = load_grid(map_path)
    grid_height, grid_width = grid.shape

    png_path = map_path.parent.parent / "sc1-png" / f"{map_path.stem}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"Map PNG not found: {png_path}")

    scenarios, runtimes, starts, goals, scenario_ids = _build_scenarios_and_runtimes(
        metadata,
        map_path=map_path,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    reward_mode = metadata.get("reward_mode", "sum")

    def env_factory():
        return StarCraftTeamEnv(
            grid=grid,
            scenarios=scenarios,
            runtimes=runtimes,
            team_name=args.team_name,
            reward_mode=reward_mode,
        )

    goal_label = policy_dir.parent.name
    agent = PPOAgent(
        env_name=env_factory,
        models_dir=str(teams_dir / args.team_name),
        goal_hypothesis=goal_label,
        episodes=int(metadata.get("episodes", args.episodes)),
        device=DEFAULT_DEVICE,
        num_envs=1,
        rollout_length=1,
        batch_size=1,
        epochs=1,
    )

    try:
        agent.load_model()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Team policy checkpoint not found under {policy_dir}"
        ) from exc

    app = QApplication(sys.argv)
    pixmap = QPixmap(str(png_path))
    if pixmap.isNull():
        raise RuntimeError(f"Failed to load PNG map from {png_path}")

    aggregate_runtime = metadata.get("aggregate_runtime") or {}
    runtime = RuntimeParameters(
        max_steps=int(aggregate_runtime.get("max_steps", 1)),
        step_penalty=float(aggregate_runtime.get("step_penalty", 0.01)),
        invalid_penalty=float(aggregate_runtime.get("invalid_penalty", 0.04)),
        goal_reward=float(aggregate_runtime.get("goal_reward", 1.0)),
        scale_used=float(aggregate_runtime.get("scale_used", 1.0)),
        progress_scale=float(aggregate_runtime.get("progress_scale", 0.1)),
        allow_diagonal_actions=bool(aggregate_runtime.get("allow_diagonal_actions", True)),
        view_mode=str(aggregate_runtime.get("view_mode", "moore")),
        view_radius=int(aggregate_runtime.get("view_radius", 1)),
    )

    runner = TeamRunner(
        agent=agent,
        env_factory=env_factory,
        deterministic=args.deterministic,
        starts=starts,
        scenario_ids=scenario_ids,
    )

    window = TeamVisualizerWindow(
        map_pixmap=pixmap,
        runner=runner,
        starts=starts,
        goals=goals,
        scenario_ids=scenario_ids,
        grid_width=grid_width,
        grid_height=grid_height,
        runtime=runtime,
        step_interval_ms=max(1, int(args.step_interval_ms)),
    )
    window.resize(900, 900)
    window.show()

    exit_code = app.exec()
    if exit_code != 0:
        QMessageBox.critical(
            None,
            "Visualizer Error",
            "Team visualization ended with a non-zero exit code.",
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
