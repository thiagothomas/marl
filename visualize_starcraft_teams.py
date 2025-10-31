#!/usr/bin/env python3
"""Visualize multiple StarCraft PPO team policies on the Moving AI map."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QTimer, QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QHBoxLayout,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from starcraft import (
    RuntimeParameters,
    Scenario,
    StarCraftScenarioEnv,
    derive_runtime_parameters,
    load_grid,
    load_runtime_from_metadata,
)
from ml.ppo import PPOAgent
from metrics.metrics import softmin


Coordinate = Tuple[int, int]

DEFAULT_TEAMS_DIR = Path("models/starcraft/teams")
DEFAULT_DEVICE = "cpu"
DEFAULT_STEP_INTERVAL_MS = 60
DEFAULT_DETERMINISTIC = False


def assign_team_colors(team_names: Sequence[str]) -> Dict[str, QColor]:
    palette = [
        "#ff6b6b",
        "#4dabf7",
        "#ffd43b",
        "#20c997",
        "#845ef7",
        "#ffa94d",
        "#e64980",
        "#63e6be",
        "#94d82d",
        "#74c0fc",
    ]
    colors: Dict[str, QColor] = {}
    for idx, team in enumerate(sorted(team_names)):
        base = QColor(palette[idx % len(palette)])
        if idx >= len(palette):
            brighten = 110 + 10 * (idx // len(palette))
            base = base.lighter(brighten)
        base.setAlpha(235)
        colors[team] = base
    return colors


@dataclass(frozen=True)
class TeamPolicy:
    team_name: str
    scenario_id: str
    map_id: str
    map_path: Path
    start: Coordinate
    goal: Coordinate
    episodes: int
    policy_dir: Path
    config_path: Path
    bucket: int
    optimal_length: float
    scenario_index: int


@dataclass(frozen=True)
class RecognitionCandidate:
    team_name: str
    scenario_id: str
    agent: PPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PPO teams for a StarCraft scenario."
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help=(
            "Scenario identifier to visualize (e.g., Archipelago_line_1250). "
            "If omitted, all discovered scenarios are animated together."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=DEFAULT_DETERMINISTIC,
        help="Replay using greedy actions instead of sampling.",
    )
    parser.add_argument(
        "--step-interval-ms",
        type=int,
        default=DEFAULT_STEP_INTERVAL_MS,
        metavar="MS",
        help="Delay between visualization steps in milliseconds.",
    )
    parser.add_argument(
        "--teams-dir",
        type=Path,
        default=DEFAULT_TEAMS_DIR,
        help="Directory containing team subdirectories.",
    )
    parser.add_argument(
        "--include-team",
        action="append",
        default=None,
        metavar="TEAM",
        help="Restrict visualization to specific team names. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-team",
        action="append",
        default=None,
        metavar="TEAM",
        help="Exclude specific team names. Can be repeated.",
    )
    return parser.parse_args()


class AgentRunner:
    """Helper that steps a trained PPO agent through the environment."""

    def __init__(
        self,
        agent: PPOAgent,
        env_factory: Callable[[], StarCraftScenarioEnv],
        *,
        deterministic: bool,
        team_name: str,
        label: str,
        goal: Coordinate,
        identifier: str,
    ) -> None:
        self._env_factory = env_factory
        self._env = env_factory()
        self._agent = agent
        self._deterministic = deterministic
        self.team_name = team_name
        self.label = label
        self.identifier = identifier

        self._obs, info = self._env.reset()
        start_position = info.get("position")
        if start_position is None:
            raise RuntimeError("Environment reset did not provide a starting position.")

        self.start: Coordinate = tuple(start_position)
        self.goal: Coordinate = tuple(goal)
        self.positions: List[Coordinate] = [self.start]
        self.rewards: List[float] = []
        self.observation_history: List[np.ndarray] = [np.array(self._obs, copy=True)]
        self.action_history: List[int] = []

        self.terminated = False
        self.truncated = False
        self.steps = 0
        self.last_reward = 0.0

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def agent(self) -> PPOAgent:
        return self._agent

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
        self.action_history.append(action)

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
        self.observation_history.append(np.array(self._obs, copy=True))
        return self.positions[-1]


class TeamsCanvas(QWidget):
    """Canvas that draws all team trajectories on the map background."""

    def __init__(
        self,
        pixmap: QPixmap,
        grid_width: int,
        grid_height: int,
        team_runners: Dict[str, List[AgentRunner]],
        team_colors: Dict[str, QColor],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._team_runners = team_runners
        self._team_colors = team_colors
        self._predictions: Dict[str, Dict[str, Any]] = {}

        self._zoom = 1.0
        self._min_zoom = 0.4
        self._max_zoom = 6.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_mouse_pos = QPointF()

    def change_zoom(self, delta: float) -> None:
        previous_zoom = self._zoom
        self._zoom = float(np.clip(self._zoom + delta, self._min_zoom, self._max_zoom))
        if self._zoom == previous_zoom:
            return
        scaling_factor = self._zoom / previous_zoom
        self._pan *= scaling_factor
        self.update()

    def set_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> None:
        self._predictions = predictions
        self.update()

    def reset_view(self) -> None:
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

    def _make_variant(self, base: QColor, idx: int, count: int) -> QColor:
        variant = QColor(base)
        factor = 100 + int(40 * (idx / max(1, count - 1)))
        variant = variant.lighter(max(100, factor))
        variant.setAlpha(220)
        return variant

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        base_scale = min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )
        scale = base_scale * self._zoom
        scaled_width = int(self._pixmap.width() * scale)
        scaled_height = int(self._pixmap.height() * scale)
        scaled = self._pixmap.scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_offset = (self.width() - scaled.width()) / 2.0 + self._pan.x()
        y_offset = (self.height() - scaled.height()) / 2.0 + self._pan.y()
        painter.drawPixmap(int(x_offset), int(y_offset), scaled)

        grid_to_pixmap_scale_x = self._pixmap.width() / self._grid_width
        grid_to_pixmap_scale_y = self._pixmap.height() / self._grid_height

        pixmap_to_canvas_scale_x = scaled.width() / self._pixmap.width()
        pixmap_to_canvas_scale_y = scaled.height() / self._pixmap.height()

        def to_canvas(pt: Coordinate) -> QPointF:
            return QPointF(
                (pt[0] + 0.5) * grid_to_pixmap_scale_x * pixmap_to_canvas_scale_x + x_offset,
                (pt[1] + 0.5) * grid_to_pixmap_scale_y * pixmap_to_canvas_scale_y + y_offset,
            )

        text_pen = QPen(QColor(240, 240, 240, 220))

        for team_idx, (team, runners) in enumerate(sorted(self._team_runners.items())):
            base_color = self._team_colors[team]
            runner_count = len(runners)
            for idx, runner in enumerate(runners):
                variant = self._make_variant(base_color, idx, runner_count)
                path = runner.positions
                if not path:
                    continue

                painter.setPen(QPen(variant, 2))
                last_point: Optional[QPointF] = None
                for coordinate in path:
                    point = to_canvas(coordinate)
                    if last_point is not None:
                        painter.drawLine(last_point, point)
                    last_point = point

                painter.setPen(QPen(variant.darker(120), 3))
                painter.setBrush(QColor(variant))
                painter.drawEllipse(to_canvas(runner.start), 5, 5)

                goal_color = QColor(variant)
                goal_color.setAlpha(200)
                painter.setPen(QPen(goal_color, 3))
                painter.setBrush(goal_color)
                painter.drawEllipse(to_canvas(runner.goal), 6, 6)

                painter.setPen(QPen(goal_color.darker(150), 2))
                painter.setBrush(goal_color.darker(150))
                painter.drawEllipse(to_canvas(path[-1]), 5, 5)

                label_point = to_canvas(runner.start)
                actual_text = f"{runner.team_name} · {runner.label}"
                painter.setPen(text_pen)
                painter.drawText(label_point + QPointF(6, -8), actual_text)

                prediction = self._predictions.get(runner.identifier)
                if prediction is not None:
                    correct = (
                        prediction["team_prediction"] == runner.team_name
                        and prediction["scenario_prediction"] == runner.label
                    )
                    pred_color = QColor("#3adb76") if correct else QColor("#ff6b6b")
                    pred_text = (
                        f"pred {prediction['team_prediction']} · {prediction['scenario_prediction']}"
                    )
                    painter.setPen(pred_color)
                    painter.drawText(label_point + QPointF(6, 8), pred_text)

        painter.end()


class TeamsOverviewWidget(QWidget):
    """Widget that animates all team agents simultaneously."""

    def __init__(
        self,
        team_runners: Dict[str, List[AgentRunner]],
        team_colors: Dict[str, QColor],
        pixmap: QPixmap,
        grid_width: int,
        grid_height: int,
        step_interval_ms: int,
        *,
        recognition_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._team_runners = team_runners
        self._team_colors = team_colors
        self._all_runners = [runner for runners in team_runners.values() for runner in runners]
        self._total_agents = len(self._all_runners)
        self._global_step = 0
        self._recognition_engine = RecognitionEngine(team_runners)
        self._recognition_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        pending_callback = recognition_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._canvas = TeamsCanvas(
            pixmap=pixmap,
            grid_width=grid_width,
            grid_height=grid_height,
            team_runners=team_runners,
            team_colors=team_colors,
        )
        layout.addWidget(self._canvas, stretch=1)

        self._status = QLabel(self)
        self._status.setStyleSheet("color: #EEEEEE; background-color: #222222; padding: 6px;")
        layout.addWidget(self._status)
        self._update_status()

        self._timer = QTimer(self)
        self._timer.setInterval(step_interval_ms)
        self._timer.timeout.connect(self._advance)
        self._timer.start()
        if pending_callback is not None:
            self.set_recognition_callback(pending_callback)
        else:
            self._emit_recognition()

    @property
    def agent_count(self) -> int:
        return self._total_agents

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
        if key in (Qt.Key.Key_R, Qt.Key.Key_0):
            self._canvas.reset_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_recognition_callback(
        self,
        callback: Optional[Callable[[List[Dict[str, Any]]], None]],
    ) -> None:
        self._recognition_callback = callback
        self._emit_recognition()

    def _emit_recognition(self) -> None:
        results = self._recognition_engine.compute_results()
        predictions = {info["identifier"]: info for info in results}
        self._canvas.set_predictions(predictions)
        if self._recognition_callback is not None:
            self._recognition_callback(results)

    def _update_status(self) -> None:
        active = sum(1 for runner in self._all_runners if not (runner.terminated or runner.truncated))
        completed = self._total_agents - active
        per_team = ", ".join(
            f"{team}: {sum(1 for r in runners if r.terminated)} / {len(runners)}"
            for team, runners in sorted(self._team_runners.items())
        )
        self._status.setText(
            f"Step {self._global_step} | Active agents: {active}/{self._total_agents} | Completed: {completed} | {per_team}"
        )

    def _advance(self) -> None:
        any_active = False
        for runner in self._all_runners:
            if runner.terminated or runner.truncated:
                continue
            runner.step()
            any_active = True
        if any_active:
            self._global_step += 1
            self._canvas.update()
            self._update_status()
            self._emit_recognition()
        else:
            self._timer.stop()
            self._update_status()
            self._emit_recognition()


class TeamsOverviewWindow(QMainWindow):
    """Main window that hosts the teams overview visualization."""

    def __init__(
        self,
        *,
        map_label: str,
        overview: TeamsOverviewWidget,
        team_colors: Dict[str, QColor],
        sidebar: "RecognitionSidebar",
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"{map_label} — Teams Overview")

        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(overview, stretch=3)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(420)
        layout.addWidget(sidebar, stretch=2)
        self.setCentralWidget(container)

        status = QStatusBar(self)
        status.setStyleSheet("color: #EEEEEE; background-color: #222222; padding: 4px;")

        legend_fragments = []
        for team, color in sorted(team_colors.items()):
            legend_fragments.append(
                f"<span style='color:{color.name()}; font-weight:600;'>■ {team}</span>"
            )
        legend_html = " | ".join(legend_fragments)

        label = QLabel(f"Teams: {legend_html}  | Agents: {overview.agent_count}")
        label.setStyleSheet("color: #EEEEEE;")
        status.addPermanentWidget(label)
        self.setStatusBar(status)


class RecognitionEngine:
    """Compute KL-divergence-based recognition assignments for observed agents."""

    def __init__(self, team_runners: Dict[str, List[AgentRunner]]) -> None:
        self._team_runners = team_runners
        self._runners: List[AgentRunner] = [
            runner for runners in team_runners.values() for runner in runners
        ]
        self._candidates: List[RecognitionCandidate] = [
            RecognitionCandidate(
                team_name=runner.team_name,
                scenario_id=runner.label,
                agent=runner.agent,
            )
            for runners in team_runners.values()
            for runner in runners
        ]
        self._epsilon = 1e-9

    def compute_results(self) -> List[Dict[str, Any]]:
        if not self._runners or not self._candidates:
            return []

        results: List[Dict[str, Any]] = []
        for runner in self._runners:
            if not runner.action_history:
                continue

            observations = runner.observation_history[:-1]
            actions = runner.action_history

            candidate_scores: List[Dict[str, Any]] = []
            for candidate in self._candidates:
                kl_values: List[float] = []
                for obs, action in zip(observations, actions):
                    hypothesis_probs = candidate.agent.get_action_probabilities(obs)
                    hypothesis_probs = np.asarray(hypothesis_probs, dtype=np.float64).reshape(-1)
                    hypothesis_probs = np.clip(hypothesis_probs, self._epsilon, 1.0)
                    hypothesis_probs = hypothesis_probs / hypothesis_probs.sum()
                    action_prob = hypothesis_probs[int(action)]
                    kl_values.append(float(-np.log(action_prob)))

                score = float(np.mean(kl_values)) if kl_values else float("inf")
                candidate_scores.append(
                    {
                        "team": candidate.team_name,
                        "scenario": candidate.scenario_id,
                        "kl": score,
                    }
                )

            if not candidate_scores:
                continue

            candidate_scores.sort(key=lambda item: item["kl"])
            best = candidate_scores[0]

            team_best_scores: Dict[str, float] = {}
            for entry in candidate_scores:
                team = entry["team"]
                score = entry["kl"]
                if team not in team_best_scores or score < team_best_scores[team]:
                    team_best_scores[team] = score

            team_names = list(team_best_scores.keys())
            team_values = np.array([team_best_scores[name] for name in team_names], dtype=np.float64)
            team_probs = softmin(team_values)
            team_probability_map = {
                name: float(prob) for name, prob in zip(team_names, team_probs)
            }

            top_hypotheses = candidate_scores[:3]
            scenario_values = np.array([entry["kl"] for entry in candidate_scores], dtype=np.float64)
            scenario_probs = softmin(scenario_values)
            goal_probability_map = {
                entry["scenario"]: float(prob) for entry, prob in zip(candidate_scores, scenario_probs)
            }

            results.append(
                {
                    "identifier": runner.identifier,
                    "agent_label": runner.label,
                    "actual_team": runner.team_name,
                    "actual_scenario": runner.label,
                    "team_prediction": best["team"],
                    "scenario_prediction": best["scenario"],
                    "score": best["kl"],
                    "team_probabilities": team_probability_map,
                    "top_hypotheses": top_hypotheses,
                    "steps_observed": len(actions),
                    "goal_probabilities": goal_probability_map,
                }
            )

        results.sort(key=lambda item: item["agent_label"])
        return results


class RecognitionSidebar(QWidget):
    """Sidebar widget that displays KL-based recognition estimates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        team_title = QLabel("Team Recognition")
        team_title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(team_title)

        self._team_table = QTableWidget(0, 5, self)
        self._team_table.setHorizontalHeaderLabels(
            [
                "Agent",
                "Actual Team",
                "Predicted Team",
                "Avg NLL",
                "Team Probabilities",
            ]
        )
        team_header = self._team_table.horizontalHeader()
        team_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        team_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        team_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        team_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        team_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._team_table.verticalHeader().setVisible(False)
        self._team_table.setAlternatingRowColors(True)
        self._team_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._team_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self._team_table, stretch=1)

        self._team_summary = QLabel("Awaiting observations…")
        self._team_summary.setStyleSheet("color: #CCCCCC; padding-top: 4px;")
        layout.addWidget(self._team_summary)

        agent_title = QLabel("Agent / Goal Recognition")
        agent_title.setStyleSheet("font-weight: 600; font-size: 15px; padding-top: 12px;")
        layout.addWidget(agent_title)

        self._agent_table = QTableWidget(0, 5, self)
        self._agent_table.setHorizontalHeaderLabels(
            [
                "Agent",
                "Actual Goal",
                "Predicted Goal",
                "Avg NLL",
                "Goal Probabilities",
            ]
        )
        agent_header = self._agent_table.horizontalHeader()
        agent_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        agent_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        agent_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        agent_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        agent_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._agent_table.verticalHeader().setVisible(False)
        self._agent_table.setAlternatingRowColors(True)
        self._agent_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._agent_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self._agent_table, stretch=1)

        self._agent_summary = QLabel("Awaiting observations…")
        self._agent_summary.setStyleSheet("color: #CCCCCC; padding-top: 4px;")
        layout.addWidget(self._agent_summary)
        layout.addStretch(1)

    def update_results(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            self._team_table.setRowCount(0)
            self._agent_table.setRowCount(0)
            self._team_summary.setText("Awaiting observations…")
            self._agent_summary.setText("Awaiting observations…")
            return

        self._team_table.setRowCount(len(results))
        self._agent_table.setRowCount(len(results))
        correct_team_predictions = 0
        correct_goal_predictions = 0
        for row, info in enumerate(results):
            agent_item = QTableWidgetItem(info["agent_label"])
            actual_team_item = QTableWidgetItem(info["actual_team"])
            predicted_team_item = QTableWidgetItem(info["team_prediction"])
            kl_item = QTableWidgetItem(f"{info['score']:.4f}")

            team_probs_sorted = sorted(
                info["team_probabilities"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
            probs_display = ", ".join(
                f"{team}:{prob:.2f}" for team, prob in team_probs_sorted[:4]
            )
            probs_item = QTableWidgetItem(probs_display)

            team_match = info["team_prediction"] == info["actual_team"]
            if team_match:
                correct_team_predictions += 1
                predicted_team_item.setForeground(QColor("#3adb76"))
            else:
                predicted_team_item.setForeground(QColor("#ff6b6b"))

            self._team_table.setItem(row, 0, agent_item)
            self._team_table.setItem(row, 1, actual_team_item)
            self._team_table.setItem(row, 2, predicted_team_item)
            self._team_table.setItem(row, 3, kl_item)
            self._team_table.setItem(row, 4, probs_item)

            agent_goal_item = QTableWidgetItem(info["agent_label"])
            actual_goal_item = QTableWidgetItem(info["actual_scenario"])
            predicted_goal_item = QTableWidgetItem(info["scenario_prediction"])
            goal_probs_sorted = sorted(
                info["goal_probabilities"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
            goal_probs_display = ", ".join(
                f"{scenario}:{prob:.2f}" for scenario, prob in goal_probs_sorted[:4]
            )
            goal_probs_item = QTableWidgetItem(goal_probs_display)

            goal_match = info["scenario_prediction"] == info["actual_scenario"]
            if goal_match:
                correct_goal_predictions += 1
                predicted_goal_item.setForeground(QColor("#3adb76"))
            else:
                predicted_goal_item.setForeground(QColor("#ff6b6b"))

            self._agent_table.setItem(row, 0, agent_goal_item)
            self._agent_table.setItem(row, 1, actual_goal_item)
            self._agent_table.setItem(row, 2, predicted_goal_item)
            self._agent_table.setItem(row, 3, QTableWidgetItem(f"{info['score']:.4f}"))
            self._agent_table.setItem(row, 4, goal_probs_item)

        team_accuracy = correct_team_predictions / max(1, len(results))
        goal_accuracy = correct_goal_predictions / max(1, len(results))
        self._team_summary.setText(
            f"Teams: {correct_team_predictions}/{len(results)} correct "
            f"({team_accuracy:.1%}) — lower NLL is better."
        )
        self._agent_summary.setText(
            f"Agents: {correct_goal_predictions}/{len(results)} goals correct "
            f"({goal_accuracy:.1%}) — lower NLL is better."
        )


def _parse_coordinate(value: Dict[str, object], *, label: str, source: Path) -> Coordinate:
    try:
        return int(value["x"]), int(value["y"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Invalid {label} coordinate in {source}")


def _parse_scenario_index(scenario_id: str) -> int:
    token = scenario_id.rsplit("_", maxsplit=1)[-1]
    if token.isdigit():
        return int(token)
    if "_line_" in scenario_id:
        suffix = scenario_id.split("_line_")[-1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def collect_team_policies(
    teams_dir: Path,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, List[TeamPolicy]]:
    scenario_map: Dict[str, List[TeamPolicy]] = {}
    include_set = {name for name in include} if include else None
    exclude_set = {name for name in exclude} if exclude else set()

    if not teams_dir.exists():
        raise FileNotFoundError(f"Teams directory not found: {teams_dir}")

    resolved_root = teams_dir.resolve()
    repo_root = resolved_root.parents[2] if len(resolved_root.parents) >= 3 else Path.cwd()

    for team_dir in sorted(path for path in teams_dir.iterdir() if path.is_dir()):
        team_name = team_dir.name
        if include_set is not None and team_name not in include_set:
            continue
        if team_name in exclude_set:
            continue
        for scenario_dir in sorted(path for path in team_dir.iterdir() if path.is_dir()):
            policy_dir = scenario_dir / "PPOAgent"
            config_path = policy_dir / "config.json"
            model_file = policy_dir / "model.pt"
            if not policy_dir.exists() or not config_path.exists() or not model_file.exists():
                continue

            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as err:
                print(f"Warning: could not parse {config_path}: {err}")
                continue

            scenario_id = str(config.get("scenario_id") or scenario_dir.name)
            map_id = config.get("map_id")
            map_path_raw = config.get("map_path")
            start_raw = config.get("start")
            goal_raw = config.get("goal")
            episodes_raw = config.get("episodes")

            if map_path_raw is None or start_raw is None or goal_raw is None or episodes_raw is None:
                print(f"Warning: missing required fields in {config_path}, skipping")
                continue

            map_path = Path(map_path_raw)
            if not map_path.is_absolute():
                map_path = (repo_root / map_path).resolve()
            if not map_path.exists():
                print(f"Warning: map file {map_path} referenced in {config_path} not found")
                continue

            try:
                start = _parse_coordinate(start_raw, label="start", source=config_path)
                goal = _parse_coordinate(goal_raw, label="goal", source=config_path)
            except ValueError as err:
                print(f"Warning: {err}")
                continue

            try:
                episodes = int(episodes_raw)
            except (TypeError, ValueError):
                print(f"Warning: invalid 'episodes' value in {config_path}, skipping")
                continue

            bucket = int(config.get("bucket", 0))
            optimal_length = float(config.get("optimal_length", 0.0))
            scenario_index = _parse_scenario_index(scenario_id)

            if not map_id:
                map_id = map_path.stem

            policy = TeamPolicy(
                team_name=team_name,
                scenario_id=scenario_id,
                map_id=str(map_id),
                map_path=map_path,
                start=start,
                goal=goal,
                episodes=episodes,
                policy_dir=policy_dir,
                config_path=config_path,
                bucket=bucket,
                optimal_length=optimal_length,
                scenario_index=scenario_index,
            )

            scenario_map.setdefault(scenario_id, []).append(policy)

    return scenario_map


def build_agent(
    team_policy: TeamPolicy,
    env_factory: Callable[[], StarCraftScenarioEnv],
) -> PPOAgent:
    temp_models_dir = team_policy.policy_dir.parent / ".viz_tmp_models" / team_policy.team_name
    temp_models_dir.mkdir(parents=True, exist_ok=True)
    agent = PPOAgent(
        env_name=env_factory,
        models_dir=str(temp_models_dir),
        goal_hypothesis=team_policy.scenario_id,
        episodes=team_policy.episodes,
        device=DEFAULT_DEVICE,
        num_envs=1,
        rollout_length=1,
        epochs=1,
        batch_size=1,
    )
    agent.model_path = str(team_policy.policy_dir)
    agent.load_model()
    return agent


def main() -> None:
    args = parse_args()

    teams_dir = Path(args.teams_dir)
    scenario_policies = collect_team_policies(
        teams_dir,
        include=args.include_team,
        exclude=args.exclude_team,
    )
    if not scenario_policies:
        raise FileNotFoundError(f"No team checkpoints discovered under {teams_dir}")

    if args.scenario_id:
        if args.scenario_id not in scenario_policies:
            available_ids = sorted(scenario_policies.keys())
            raise ValueError(
                f"Scenario '{args.scenario_id}' not found in {teams_dir}. "
                f"Available scenarios: {', '.join(available_ids)}"
            )
        active_policies_map = {args.scenario_id: scenario_policies[args.scenario_id]}
    else:
        active_policies_map = scenario_policies

    policies: List[TeamPolicy] = [
        policy for items in active_policies_map.values() for policy in items
    ]
    if not policies:
        raise FileNotFoundError(f"No team checkpoints found under {teams_dir}")

    map_paths = {policy.map_path for policy in policies}
    if len(map_paths) != 1:
        maps_display = ", ".join(sorted(str(path) for path in map_paths))
        raise ValueError(
            "Team checkpoints span multiple map files. "
            "Filter with --scenario-id or reorganize teams. "
            f"Found maps: {maps_display}"
        )
    map_path = next(iter(map_paths))

    grid = load_grid(map_path)
    grid_height, grid_width = grid.shape

    png_path = map_path.parent.parent / "sc1-png" / f"{map_path.stem}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"Map PNG not found: {png_path}")

    app = QApplication(sys.argv)
    pixmap = QPixmap(str(png_path))
    if pixmap.isNull():
        raise RuntimeError(f"Failed to load PNG map from {png_path}")

    teams: Dict[str, List[TeamPolicy]] = {}
    for policy in policies:
        teams.setdefault(policy.team_name, []).append(policy)

    team_colors = assign_team_colors(sorted(teams.keys()))

    team_runners: Dict[str, List[AgentRunner]] = {}
    agent_counter = 0
    for team_name, team_policies in sorted(teams.items()):
        team_runners[team_name] = []
        for policy in sorted(team_policies, key=lambda p: (p.bucket, p.scenario_index, p.scenario_id)):
            scenario = Scenario(
                map_id=policy.map_id,
                map_path=policy.map_path,
                bucket=policy.bucket,
                width=grid_width,
                height=grid_height,
                start=policy.start,
                goal=policy.goal,
                optimal_length=policy.optimal_length,
                index=policy.scenario_index,
            )

            runtime = load_runtime_from_metadata(policy.config_path, scenario)
            if runtime is None:
                runtime = derive_runtime_parameters(
                    scenario,
                    None,
                    allow_diagonal_actions=True,
                    view_mode="moore",
                    view_radius=1,
                )

            def env_factory(
                scenario=scenario,
                runtime=runtime,
            ) -> StarCraftScenarioEnv:
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

            agent = build_agent(policy, env_factory)
            identifier = f"{team_name}|{policy.scenario_id}|{agent_counter}"
            agent_counter += 1
            runner = AgentRunner(
                agent=agent,
                env_factory=env_factory,
                deterministic=args.deterministic,
                team_name=team_name,
                label=policy.scenario_id,
                goal=policy.goal,
                identifier=identifier,
            )
            team_runners[team_name].append(runner)

    map_label_candidates = {policy.map_id for policy in policies}
    map_label = next(iter(map_label_candidates)) if len(map_label_candidates) == 1 else map_path.stem

    sidebar = RecognitionSidebar()

    overview = TeamsOverviewWidget(
        team_runners=team_runners,
        team_colors=team_colors,
        pixmap=pixmap,
        grid_width=grid_width,
        grid_height=grid_height,
        step_interval_ms=max(1, int(args.step_interval_ms)),
        recognition_callback=sidebar.update_results,
    )
    overview.setMinimumWidth(580)
    overview.set_recognition_callback(sidebar.update_results)

    window = TeamsOverviewWindow(
        map_label=map_label,
        overview=overview,
        team_colors=team_colors,
        sidebar=sidebar,
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
