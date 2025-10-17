#!/usr/bin/env python3
"""
Modern GUI Visualizer for Multi-Agent Goal Recognition.

A clean, aesthetic interface built with PyQt6 for visualizing the recognition process
with real-time updates, metrics, and predictions.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations, product

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QScrollArea, QComboBox, QSpinBox, QProgressBar, QDialog,
    QDialogButtonBox, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPalette

from envs.team_goal_environments import (
    TeamGoalTopRight, TeamGoalTopLeft, TeamGoalBottomLeft,
    TeamGoalBottomRight
)
from envs.multi_agent_grid_world import MultiAgentGridWorld
from ml.ppo import PPOAgent
from metrics.metrics import kl_divergence


class GridCanvas(QWidget):
    """Canvas for rendering the grid world with agents and trajectories."""

    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = 60
        self.agent_positions = {}
        self.trajectories = {}
        self.goal_positions = {}
        self.team_goals = []
        self.agent_teams = []
        self.goals_reached = {}  # Track which agents reached goals
        self.obstacles = []
        self.setMinimumSize(self.grid_size * self.cell_size + 50,
                           self.grid_size * self.cell_size + 50)

        # Modern color scheme
        self.colors = {
            'grid': QColor(240, 240, 245),
            'grid_line': QColor(200, 200, 210),
            'path': QColor(100, 150, 255, 60),
            'teams': [
                QColor(255, 107, 107),  # Red
                QColor(78, 205, 196),   # Teal
                QColor(255, 195, 113),  # Orange
                QColor(162, 155, 254),  # Purple
            ],
            'goal': QColor(255, 215, 0, 100),  # Gold
            'obstacle': QColor(120, 130, 150)  # Neutral slate
        }

    def update_state(self, agent_positions, trajectories, goal_positions,
                    team_goals, agent_teams, goals_reached=None, obstacles=None):
        """Update the grid state."""
        self.agent_positions = agent_positions
        self.trajectories = trajectories
        self.goal_positions = goal_positions
        self.team_goals = team_goals
        self.agent_teams = agent_teams
        self.goals_reached = goals_reached if goals_reached is not None else {}
        self.obstacles = obstacles if obstacles is not None else []
        self.update()

    def paintEvent(self, event):
        """Render the grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        offset_x, offset_y = 25, 25

        # Draw grid background
        painter.fillRect(offset_x, offset_y,
                        self.grid_size * self.cell_size,
                        self.grid_size * self.cell_size,
                        self.colors['grid'])

        # Draw obstacles before grid lines to appear under outlines
        if self.obstacles:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.colors['obstacle']))
            for ox, oy in self.obstacles:
                painter.drawRect(
                    offset_x + ox * self.cell_size,
                    offset_y + oy * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

        # Draw grid lines
        painter.setPen(QPen(self.colors['grid_line'], 1))
        for i in range(self.grid_size + 1):
            # Vertical lines
            painter.drawLine(offset_x + i * self.cell_size, offset_y,
                           offset_x + i * self.cell_size,
                           offset_y + self.grid_size * self.cell_size)
            # Horizontal lines
            painter.drawLine(offset_x, offset_y + i * self.cell_size,
                           offset_x + self.grid_size * self.cell_size,
                           offset_y + i * self.cell_size)

        # Draw goal positions (draw them first, so agents appear on top)
        for team_id, goal_pos in self.goal_positions.items():
            x, y = int(goal_pos[0]), int(goal_pos[1])
            painter.fillRect(
                offset_x + x * self.cell_size + 2,
                offset_y + y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4,
                self.colors['goal']
            )

        # Draw trajectories
        for agent_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue

            agent_idx = int(agent_id.split('_')[1])
            team_id = self.agent_teams[agent_idx] if agent_idx < len(self.agent_teams) else 0
            color = self.colors['teams'][team_id % len(self.colors['teams'])]

            # Draw path as connected lines
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 80), 3))
            for i in range(len(trajectory) - 1):
                pos1, _ = trajectory[i]
                pos2, _ = trajectory[i + 1]
                x1, y1 = int(pos1[0]), int(pos1[1])
                x2, y2 = int(pos2[0]), int(pos2[1])

                painter.drawLine(
                    offset_x + x1 * self.cell_size + self.cell_size // 2,
                    offset_y + y1 * self.cell_size + self.cell_size // 2,
                    offset_x + x2 * self.cell_size + self.cell_size // 2,
                    offset_y + y2 * self.cell_size + self.cell_size // 2
                )

        # Draw agents
        for agent_id, pos in self.agent_positions.items():
            agent_idx = int(agent_id.split('_')[1])
            team_id = self.agent_teams[agent_idx] if agent_idx < len(self.agent_teams) else 0
            color = self.colors['teams'][team_id % len(self.colors['teams'])]

            x, y = int(pos[0]), int(pos[1])
            center_x = offset_x + x * self.cell_size + self.cell_size // 2
            center_y = offset_y + y * self.cell_size + self.cell_size // 2

            # Check if agent has reached goal
            has_reached_goal = self.goals_reached.get(agent_id, False)

            # Draw agent circle with border if goal reached
            painter.setBrush(QBrush(color))
            if has_reached_goal:
                # Green border for reached goal
                painter.setPen(QPen(QColor(46, 213, 115), 4))
            else:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(QPointF(center_x, center_y), 18, 18)

            # Draw agent number or checkmark
            painter.setPen(QPen(QColor(255, 255, 255)))
            font = QFont('Arial', 12, QFont.Weight.Bold)
            painter.setFont(font)
            if has_reached_goal:
                # Draw checkmark
                painter.drawText(center_x - 6, center_y + 6, "✓")
            else:
                painter.drawText(center_x - 6, center_y + 6, str(agent_idx))

        # Draw goal labels AFTER agents so they don't obscure agents
        for team_id, goal_pos in self.goal_positions.items():
            x, y = int(goal_pos[0]), int(goal_pos[1])
            # Draw goal label slightly offset if there's an agent at the goal
            has_agent_at_goal = any(
                np.array_equal(pos, goal_pos) for pos in self.agent_positions.values()
            )
            painter.setPen(QPen(QColor(180, 150, 0), 2))
            font = QFont('Arial', 10, QFont.Weight.Bold)
            painter.setFont(font)
            label_y_offset = 45 if has_agent_at_goal else 20  # Move label down if agent is there
            painter.drawText(
                offset_x + x * self.cell_size + self.cell_size // 2 - 10,
                offset_y + y * self.cell_size + label_y_offset,
                f"G{team_id}"
            )


class ConfidencePanel(QWidget):
    """Panel for visualizing agent-goal confidence with progress bars."""

    def __init__(self, num_agents, goal_names):
        super().__init__()
        self.num_agents = num_agents
        self.goal_names = goal_names
        self.agent_scores = {}  # Store scores per agent per goal
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Goal Confidence Per Agent")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
                background-color: white;
                border-radius: 5px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Scroll area for agent confidence displays
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # Create confidence display for each agent
        self.agent_widgets = {}
        for agent_idx in range(self.num_agents):
            agent_id = f'agent_{agent_idx}'
            agent_widget = self.create_agent_confidence_widget(agent_id)
            scroll_layout.addWidget(agent_widget)
            self.agent_widgets[agent_id] = agent_widget

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self.setMinimumWidth(300)

    def create_agent_confidence_widget(self, agent_id):
        """Create a widget showing confidence bars for one agent."""
        group = QGroupBox(f"Agent {agent_id.split('_')[1]}")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: white;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Store progress bars for this agent
        progress_bars = {}

        # Create a progress bar for each goal
        for goal_name in self.goal_names:
            goal_short = goal_name.replace('team_goal_', '')

            row = QHBoxLayout()
            row.setSpacing(8)

            # Goal label
            label = QLabel(f"{goal_short}")
            label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    color: #34495e;
                    min-width: 90px;
                }
            """)
            row.addWidget(label)

            # Progress bar
            pbar = QProgressBar()
            pbar.setRange(0, 1000)  # Use 0-1000 for better precision
            pbar.setValue(0)
            pbar.setTextVisible(True)
            pbar.setFormat("%p%")
            pbar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    text-align: center;
                    height: 18px;
                    background-color: #ecf0f1;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 2px;
                }
            """)
            pbar.setMinimumWidth(150)
            row.addWidget(pbar, stretch=1)

            # Percentage label
            pct_label = QLabel("0.0%")
            pct_label.setStyleSheet("""
                QLabel {
                    font-size: 10px;
                    color: #7f8c8d;
                    min-width: 45px;
                }
            """)
            row.addWidget(pct_label)

            layout.addLayout(row)
            progress_bars[goal_name] = (pbar, pct_label, label)

        # Store reference to progress bars
        group.progress_bars = progress_bars
        group.setLayout(layout)

        return group

    def update_confidence(self, agent_scores, true_assignment):
        """Update confidence displays for all agents.

        agent_scores: dict[agent_id] -> dict[goal_name] -> score
        true_assignment: dict[agent_id] -> (team_id, goal_name)
        """
        for agent_id, agent_widget in self.agent_widgets.items():
            if agent_id not in agent_scores:
                continue

            scores = agent_scores[agent_id]
            if not scores:
                continue

            # Normalize scores to percentages (softmax-like)
            score_values = list(scores.values())
            min_score = min(score_values)
            max_score = max(score_values)

            # Shift scores to be positive for percentage calculation
            shifted_scores = {g: s - min_score + 0.01 for g, s in scores.items()}
            total = sum(shifted_scores.values())

            true_goal = true_assignment.get(agent_id, (None, None))[1]

            # Update each progress bar
            for goal_name, (pbar, pct_label, label) in agent_widget.progress_bars.items():
                score = scores.get(goal_name, 0.0)
                percentage = (shifted_scores[goal_name] / total) * 100 if total > 0 else 0

                # Update progress bar value (0-1000 range)
                pbar.setValue(int(percentage * 10))

                # Update percentage label
                pct_label.setText(f"{percentage:.1f}%")

                # Highlight the true goal
                is_true_goal = (goal_name == true_goal)

                if is_true_goal:
                    # Green for true goal
                    pbar.setStyleSheet("""
                        QProgressBar {
                            border: 2px solid #27ae60;
                            border-radius: 3px;
                            text-align: center;
                            height: 18px;
                            background-color: #ecf0f1;
                            font-size: 10px;
                        }
                        QProgressBar::chunk {
                            background-color: #2ecc71;
                            border-radius: 2px;
                        }
                    """)
                    label.setStyleSheet("""
                        QLabel {
                            font-size: 11px;
                            color: #27ae60;
                            font-weight: bold;
                            min-width: 90px;
                        }
                    """)
                    # Add checkmark
                    if "✓" not in label.text():
                        goal_short = goal_name.replace('team_goal_', '')
                        label.setText(f"{goal_short} ✓")
                else:
                    # Blue for other goals
                    pbar.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #bdc3c7;
                            border-radius: 3px;
                            text-align: center;
                            height: 18px;
                            background-color: #ecf0f1;
                            font-size: 10px;
                        }
                        QProgressBar::chunk {
                            background-color: #3498db;
                            border-radius: 2px;
                        }
                    """)
                    label.setStyleSheet("""
                        QLabel {
                            font-size: 11px;
                            color: #34495e;
                            min-width: 90px;
                        }
                    """)


class MetricsPanel(QWidget):
    """Panel for displaying recognition metrics and predictions."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Title
        title = QLabel("Recognition Metrics")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        layout.addWidget(title)

        # Predictions table
        predictions_group = QGroupBox("Top Hypotheses")
        predictions_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: white;
            }
        """)

        pred_layout = QVBoxLayout()
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(4)
        self.predictions_table.setHorizontalHeaderLabels(
            ["Rank", "Score", "Assignment", "Correct"]
        )
        self.predictions_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 5px;
                gridline-color: #ecf0f1;
                color: #2c3e50;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        pred_layout.addWidget(self.predictions_table)
        predictions_group.setLayout(pred_layout)
        layout.addWidget(predictions_group)

        # Observations info
        obs_group = QGroupBox("Observations")
        obs_group.setStyleSheet(predictions_group.styleSheet())
        obs_layout = QVBoxLayout()
        self.obs_label = QLabel("No observations collected yet")
        self.obs_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                color: #2c3e50;
            }
        """)
        obs_layout.addWidget(self.obs_label)
        obs_group.setLayout(obs_layout)
        layout.addWidget(obs_group)

        # Accuracy metrics
        acc_group = QGroupBox("Accuracy Metrics")
        acc_group.setStyleSheet(predictions_group.styleSheet())
        acc_layout = QVBoxLayout()

        self.goal_accuracy_label = QLabel("Goal Accuracy: --")
        self.team_accuracy_label = QLabel("Team Accuracy: --")

        for label in [self.goal_accuracy_label, self.team_accuracy_label]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 13px;
                    padding: 5px;
                    color: #34495e;
                }
            """)
            acc_layout.addWidget(label)

        acc_group.setLayout(acc_layout)
        layout.addWidget(acc_group)

        # Latency metrics
        latency_group = QGroupBox("Recognition Latency")
        latency_group.setStyleSheet(predictions_group.styleSheet())
        latency_layout = QVBoxLayout()

        self.goal_latency_label = QLabel("Goal lock-in: --")
        self.team_latency_label = QLabel("Team lock-in: --")
        self.joint_latency_label = QLabel("Joint lock-in: --")

        for label in [self.goal_latency_label, self.team_latency_label, self.joint_latency_label]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    padding: 5px;
                    color: #2c3e50;
                }
            """)
            latency_layout.addWidget(label)

        latency_group.setLayout(latency_layout)
        layout.addWidget(latency_group)

        layout.addStretch()
        self.setLayout(layout)

    def update_predictions(self, best_assignments, true_assignment, top_k=5):
        """Update the predictions table."""
        self.predictions_table.setRowCount(min(top_k, len(best_assignments)))

        for rank, assignment_info in enumerate(best_assignments[:top_k]):
            assignment = assignment_info['assignment']
            score = assignment_info['score']

            # Check if correct
            is_correct = all(
                assignment.get(aid, (None, None)) == true_assignment.get(aid, (None, None))
                for aid in assignment.keys()
            )

            # Rank
            rank_item = QTableWidgetItem(f"#{rank + 1}")
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_correct:
                rank_item.setBackground(QColor(46, 213, 115, 100))

            # Score with breakdown tooltip
            score_item = QTableWidgetItem(f"{score:.2f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Add tooltip showing individual scores
            score_details = assignment_info.get('score_details', [])
            if score_details:
                score_item.setToolTip("\\n".join(score_details))

            # Assignment
            teams_str = []
            teams_display = {}
            for agent_id, (team_id, goal) in assignment.items():
                if team_id not in teams_display:
                    teams_display[team_id] = []
                agent_num = agent_id.split('_')[1]
                teams_display[team_id].append((agent_num, goal))

            for team_id in sorted(teams_display.keys()):
                agents_goals = teams_display[team_id]
                agent_nums = [a for a, _ in agents_goals]
                goal = agents_goals[0][1].replace('team_goal_', '')
                teams_str.append(f"T{team_id}[{','.join(agent_nums)}]→{goal}")

            assignment_item = QTableWidgetItem(" | ".join(teams_str))

            # Correct marker
            correct_item = QTableWidgetItem("✓" if is_correct else "")
            correct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_correct:
                correct_item.setForeground(QColor(46, 213, 115))
                font = correct_item.font()
                font.setPointSize(14)
                correct_item.setFont(font)

            self.predictions_table.setItem(rank, 0, rank_item)
            self.predictions_table.setItem(rank, 1, score_item)
            self.predictions_table.setItem(rank, 2, assignment_item)
            self.predictions_table.setItem(rank, 3, correct_item)

        self.predictions_table.resizeColumnsToContents()

    def update_observations(self, observations_per_agent):
        """Update observation counts."""
        obs_text = ""
        for agent_id in sorted(observations_per_agent.keys()):
            count = len(observations_per_agent[agent_id])
            obs_text += f"{agent_id}: {count} observations\n"

            # Show last observation details for debugging
            if count > 0:
                last_obs = observations_per_agent[agent_id][-1]
                pos, action = last_obs
                action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                obs_text += f"  Last: pos=({int(pos[0])},{int(pos[1])}), action={action_names[action]}\n"

        self.obs_label.setText(obs_text.strip())

    def update_accuracy(self, best_assignment, true_assignment, num_agents):
        """Update accuracy metrics."""
        correct_goals = sum(
            1 for aid in best_assignment.keys()
            if best_assignment[aid][1] == true_assignment[aid][1]
        )
        correct_teams = sum(
            1 for aid in best_assignment.keys()
            if best_assignment[aid][0] == true_assignment[aid][0]
        )

        goal_acc = correct_goals / num_agents
        team_acc = correct_teams / num_agents

        self.goal_accuracy_label.setText(
            f"Goal Accuracy: {goal_acc:.1%} ({correct_goals}/{num_agents})"
        )
        self.team_accuracy_label.setText(
            f"Team Accuracy: {team_acc:.1%} ({correct_teams}/{num_agents})"
        )

    def update_latency(self, latency_info):
        """Update latency labels."""
        if not latency_info or not latency_info.get('max_observations'):
            self.goal_latency_label.setText("Goal lock-in: --")
            self.team_latency_label.setText("Team lock-in: --")
            self.joint_latency_label.setText("Joint lock-in: --")
            return

        max_obs = latency_info['max_observations']

        def fmt(value):
            return f"{value}/{max_obs}" if value is not None else f"not reached ≤ {max_obs}"

        self.goal_latency_label.setText(f"Goal lock-in: {fmt(latency_info.get('goal_latency'))}")
        self.team_latency_label.setText(f"Team lock-in: {fmt(latency_info.get('team_latency'))}")
        self.joint_latency_label.setText(f"Joint lock-in: {fmt(latency_info.get('joint_latency'))}")


class VisualizerMainWindow(QMainWindow):
    """Main window for the visualization application."""

    def __init__(self, agents, goal_names, num_agents, team_sizes, team_goals):
        super().__init__()
        self.agents = agents
        self.goal_names = goal_names
        self.num_agents = num_agents
        self.team_sizes = team_sizes
        self.team_goals = team_goals
        self.num_teams = len(team_sizes)

        # State
        self.env = None
        self.current_step = 0
        self.max_steps = 30
        self.observations_per_agent = {}
        self.trajectories = {}
        self.current_positions = {}
        self.true_assignment = {}
        self.obs_dict = None
        self.is_running = False
        self.goals_reached = {}  # Track which agents have reached their goals

        self.init_ui()
        self.init_environment()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Multi-Agent Goal Recognition Visualizer")
        self.setGeometry(100, 100, 1400, 900)

        # Apply modern styling (removed QPushButton global style - using per-button styles instead)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Multi-Agent Goal Recognition System")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 15px;
                background-color: white;
                border-radius: 8px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.max_steps)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Main content: Grid + Confidence + Metrics
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Grid canvas
        grid_container = QWidget()
        grid_layout = QVBoxLayout(grid_container)
        grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        grid_label = QLabel("Environment")
        grid_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
            }
        """)
        grid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(grid_label)

        self.grid_canvas = GridCanvas(grid_size=7)
        grid_layout.addWidget(self.grid_canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        # Confidence panel (middle)
        self.confidence_panel = ConfidencePanel(self.num_agents, self.goal_names)

        # Metrics panel (right)
        self.metrics_panel = MetricsPanel()

        content_splitter.addWidget(grid_container)
        content_splitter.addWidget(self.confidence_panel)
        content_splitter.addWidget(self.metrics_panel)
        content_splitter.setStretchFactor(0, 2)  # Grid gets more space
        content_splitter.setStretchFactor(1, 1)  # Confidence panel
        content_splitter.setStretchFactor(2, 1)  # Metrics panel

        main_layout.addWidget(content_splitter, stretch=1)

        # Status bar
        self.status_label = QLabel("Ready to start")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                padding: 8px;
                background-color: white;
                border-radius: 5px;
                color: #7f8c8d;
            }
        """)
        main_layout.addWidget(self.status_label)

    def create_control_panel(self):
        """Create the control panel with buttons."""
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        layout = QHBoxLayout(panel)

        # Button styling - explicit and clear
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: 2px solid #2980b9;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                min-width: 120px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border: 2px solid #21618c;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
                border: 2px solid #95a5a6;
            }
        """

        # Buttons with clear text (no emojis in case of font issues)
        self.start_btn = QPushButton("START")
        self.start_btn.setStyleSheet(button_style)
        self.start_btn.clicked.connect(self.start_simulation)

        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.setStyleSheet(button_style)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.pause_btn.setEnabled(False)

        self.step_btn = QPushButton("STEP")
        self.step_btn.setStyleSheet(button_style)
        self.step_btn.clicked.connect(self.step_simulation)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.setStyleSheet(button_style)
        self.reset_btn.clicked.connect(self.reset_simulation)

        for btn in [self.start_btn, self.pause_btn, self.step_btn, self.reset_btn]:
            layout.addWidget(btn)

        layout.addStretch()

        # Speed control
        speed_label = QLabel("Speed (ms):")
        speed_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                font-size: 14px;
                padding: 5px;
            }
        """)
        layout.addWidget(speed_label)

        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(100, 2000)
        self.speed_spinbox.setValue(500)
        self.speed_spinbox.setSingleStep(100)
        self.speed_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
                font-size: 13px;
                min-width: 80px;
                min-height: 35px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                border: 1px solid #bdc3c7;
            }
        """)
        layout.addWidget(self.speed_spinbox)

        return panel

    def init_environment(self):
        """Initialize the environment and state."""
        self.env = MultiAgentGridWorld(
            size=7,
            max_steps=100,
            team_sizes=self.team_sizes,
            team_goals=self.team_goals,
            render_mode=None
        )

        self.observations_per_agent = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.trajectories = {f'agent_{i}': [] for i in range(self.num_agents)}
        self.goals_reached = {f'agent_{i}': False for i in range(self.num_agents)}

        self.true_assignment = {}

        # Define true assignment
        agent_counter = 0
        for team_id, team_size in enumerate(self.team_sizes):
            team_goal_str = f"team_goal_{self.team_goals[team_id]}"
            for _ in range(team_size):
                self.true_assignment[f'agent_{agent_counter}'] = (team_id, team_goal_str)
                agent_counter += 1

        # Reset environment
        self.obs_dict, info = self.env.reset()
        self.initial_positions = info.get('initial_positions')

        # Extract initial positions
        self.current_positions = {}
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            pos_offset = i * 2
            self.current_positions[agent_id] = self.obs_dict[agent_id][pos_offset:pos_offset+2].copy()

        if hasattr(self, "status_label"):
            if self.initial_positions:
                pos_summary = ", ".join(
                    f"agent_{idx}:{tuple(pos)}" for idx, pos in enumerate(self.initial_positions)
                )
                self.status_label.setText(f"Ready to start | initial positions: {pos_summary}")
            else:
                self.status_label.setText("Ready to start")

        # Update display
        self.update_display()

        # Timer for automatic stepping
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)

    def start_simulation(self):
        """Start the simulation."""
        if not self.is_running:
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.timer.start(self.speed_spinbox.value())
            self.status_label.setText("Running...")

    def pause_simulation(self):
        """Pause the simulation."""
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.status_label.setText("Paused")

    def step_simulation(self):
        """Execute one step of the simulation."""
        if self.current_step >= self.max_steps:
            self.pause_simulation()
            self.status_label.setText("Simulation complete!")
            return

        # Create expert policies
        actions = {}
        for agent_id in self.observations_per_agent.keys():
            agent_idx = int(agent_id.split('_')[1])
            team_id = self.env.agent_teams[agent_idx]
            team_goal = self.team_goals[team_id]
            goal_pos = self.env.goal_positions[team_id]

            # Check if agent has reached goal
            pos_offset = agent_idx * 2
            agent_pos = self.obs_dict[agent_id][pos_offset:pos_offset+2].copy()

            if np.array_equal(agent_pos, goal_pos):
                self.goals_reached[agent_id] = True
                # Don't move if at goal - stay in place
                action = 0  # UP action (but won't move if at goal)
            else:
                action = self.expert_policy(self.obs_dict[agent_id], team_goal, agent_idx)

            actions[agent_id] = action

            # Store observation only if not at goal
            if not self.goals_reached[agent_id]:
                self.observations_per_agent[agent_id].append((agent_pos, action))
                self.trajectories[agent_id].append((agent_pos, action))

        # Step environment
        self.obs_dict, rewards, terms, truncs, info = self.env.step(actions)

        # Update positions
        for agent_id in self.current_positions.keys():
            agent_idx = int(agent_id.split('_')[1])
            pos_offset = agent_idx * 2
            self.current_positions[agent_id] = self.obs_dict[agent_id][pos_offset:pos_offset+2].copy()

        self.current_step += 1
        self.progress_bar.setValue(self.current_step)

        # Update display
        self.update_display()

        # Check if all agents have reached their goals AFTER updating positions
        if all(self.goals_reached.values()):
            self.pause_simulation()
            # Disable step button when complete
            self.step_btn.setEnabled(False)
            self.start_btn.setEnabled(False)

            # Set progress bar to maximum to show completion
            self.progress_bar.setValue(self.max_steps)

            # Show completion message with recognized assignments
            best_assignments = self.compute_team_assignment_scores()
            if best_assignments:
                best = best_assignments[0]['assignment']
                msg_parts = ["✓ Recognition Complete!"]
                for agent_id in sorted(best.keys()):
                    team_id, goal = best[agent_id]
                    goal_short = goal.replace('team_goal_', '')
                    msg_parts.append(f"{agent_id}→Team{team_id}→{goal_short}")
                self.status_label.setText("  |  ".join(msg_parts))
            return

        # Update status to show which agents have reached goals
        reached_count = sum(1 for reached in self.goals_reached.values() if reached)
        status_parts = [f"{reached_count}/{self.num_agents} goals reached"]
        for agent_id in sorted(self.goals_reached.keys()):
            if self.goals_reached[agent_id]:
                status_parts.append(f"{agent_id}: ✓")
        self.status_label.setText(" | ".join(status_parts))

        # Check if done
        if all(terms.values()) or all(truncs.values()):
            self.pause_simulation()
            self.status_label.setText(f"Episode finished! Team success: {info['team_success']}")

    def reset_simulation(self):
        """Reset the simulation."""
        self.pause_simulation()
        self.current_step = 0
        self.progress_bar.setValue(0)
        # Re-enable buttons
        self.step_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.init_environment()
        if self.initial_positions:
            pos_summary = ", ".join(
                f"agent_{idx}:{tuple(pos)}" for idx, pos in enumerate(self.initial_positions)
            )
            self.status_label.setText(f"Reset - Ready | initial positions: {pos_summary}")
        else:
            self.status_label.setText("Reset - Ready to start")

    def update_display(self):
        """Update all display elements."""
        # Update grid
        self.grid_canvas.update_state(
            self.current_positions,
            self.trajectories,
            self.env.goal_positions,
            self.env.team_goals,
            self.env.agent_teams,
            self.goals_reached,
            obstacles=self.env.obstacles
        )

        # Update observations
        self.metrics_panel.update_observations(self.observations_per_agent)

        # Compute and update predictions if we have observations
        if self.current_step > 0:
            # Compute per-agent goal scores for confidence panel
            agent_scores = self.compute_per_agent_goal_scores()
            self.confidence_panel.update_confidence(agent_scores, self.true_assignment)

            # Compute team assignments
            best_assignments = self.compute_team_assignment_scores()
            if best_assignments:
                self.metrics_panel.update_predictions(
                    best_assignments, self.true_assignment, top_k=5
                )
                self.metrics_panel.update_accuracy(
                    best_assignments[0]['assignment'],
                    self.true_assignment,
                    self.num_agents
                )
                latency_info = self.compute_latency_metrics()
                self.metrics_panel.update_latency(latency_info)
        else:
            self.metrics_panel.update_latency({})

    def compute_per_agent_goal_scores(self):
        """Compute KL divergence scores for each agent against each goal policy.

        Returns:
            dict[agent_id] -> dict[goal_name] -> score
        """
        agent_scores = {}

        for agent_id in self.observations_per_agent.keys():
            if len(self.observations_per_agent[agent_id]) == 0:
                continue

            agent_obs = self.observations_per_agent[agent_id]
            goal_scores = {}

            for goal_idx, goal_name in enumerate(self.goal_names):
                # Compute KL divergence (lower is better)
                kl_div = kl_divergence(agent_obs, self.agents[goal_idx])
                # Convert to score (higher is better)
                score = -kl_div
                goal_scores[goal_name] = score

            agent_scores[agent_id] = goal_scores

        return agent_scores

    def _generate_goal_combinations(self):
        """Generate all possible goal combinations for the configured teams."""
        return list(product(self.goal_names, repeat=self.num_teams))

    def _score_team_assignments(self, observations_per_agent, possible_team_goals, team_partitions):
        ranked = []

        for partition in team_partitions:
            for goal_combination in possible_team_goals:
                if len(goal_combination) != len(partition):
                    continue

                total_score = 0.0
                assignment = {}
                score_details = []
                valid_combo = True

                for team_id, team_agents in enumerate(partition):
                    team_goal = goal_combination[team_id]
                    try:
                        goal_idx = self.goal_names.index(team_goal)
                    except ValueError:
                        valid_combo = False
                        break

                    agent_policy = self.agents[goal_idx]

                    for agent_idx in team_agents:
                        agent_id = f'agent_{agent_idx}'
                        agent_obs = observations_per_agent.get(agent_id, [])

                        if agent_obs:
                            kl_div = kl_divergence(agent_obs, agent_policy)
                            score = -kl_div
                            total_score += score
                            score_details.append(f"{agent_id}→{team_goal}: {score:.4f}")

                        assignment[agent_id] = (team_id, team_goal)

                if not valid_combo:
                    continue

                ranked.append({
                    'score': total_score,
                    'assignment': assignment,
                    'partition': partition,
                    'goals': goal_combination,
                    'score_details': score_details
                })

        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def compute_team_assignment_scores(self, possible_team_goals=None):
        """Compute scores for all possible team assignments."""
        agent_indices = list(range(self.num_agents))
        all_partitions = self._generate_team_partitions(agent_indices, self.team_sizes)

        if possible_team_goals is None:
            possible_team_goals = self._generate_goal_combinations()

        return self._score_team_assignments(
            self.observations_per_agent,
            possible_team_goals,
            all_partitions
        )

    def compute_latency_metrics(self, possible_team_goals=None):
        """Compute latency information for recognition."""
        if possible_team_goals is None:
            possible_team_goals = self._generate_goal_combinations()

        per_agent_counts = {aid: len(obs) for aid, obs in self.observations_per_agent.items()}
        max_observations = max(per_agent_counts.values()) if per_agent_counts else 0
        team_partitions = self._generate_team_partitions(list(range(self.num_agents)), self.team_sizes)

        if not self.true_assignment:
            return {
                'goal_latency': None,
                'team_latency': None,
                'joint_latency': None,
                'max_observations': max_observations,
                'per_agent_counts': per_agent_counts
            }

        goal_latency = None
        team_latency = None
        joint_latency = None

        if max_observations == 0:
            return {
                'goal_latency': None,
                'team_latency': None,
                'joint_latency': None,
                'max_observations': 0,
                'per_agent_counts': per_agent_counts
            }

        for step in range(1, max_observations + 1):
            truncated = {
                aid: (obs if len(obs) <= step else obs[:step])
                for aid, obs in self.observations_per_agent.items()
            }

            step_ranked = self._score_team_assignments(
                truncated,
                possible_team_goals,
                team_partitions
            )
            if not step_ranked:
                continue

            step_best = step_ranked[0]['assignment']

            goal_matches = sum(
                1 for aid in self.true_assignment.keys()
                if step_best.get(aid, (None, None))[1] == self.true_assignment[aid][1]
            )
            team_matches = sum(
                1 for aid in self.true_assignment.keys()
                if step_best.get(aid, (None, None))[0] == self.true_assignment[aid][0]
            )
            joint_matches = sum(
                1 for aid in self.true_assignment.keys()
                if step_best.get(aid) == self.true_assignment[aid]
            )

            if goal_latency is None and goal_matches == self.num_agents:
                goal_latency = step
            if team_latency is None and team_matches == self.num_agents:
                team_latency = step
            if joint_latency is None and joint_matches == self.num_agents:
                joint_latency = step

            if goal_latency and team_latency and joint_latency:
                break

        return {
            'goal_latency': goal_latency,
            'team_latency': team_latency,
            'joint_latency': joint_latency,
            'max_observations': max_observations,
            'per_agent_counts': per_agent_counts
        }

    def _generate_team_partitions(self, agents, team_sizes):
        """Generate all possible ways to partition agents into teams."""
        if len(team_sizes) == 0:
            return [[]]
        if len(team_sizes) == 1:
            return [[agents]]

        partitions = []
        first_team_size = team_sizes[0]
        remaining_sizes = team_sizes[1:]

        for first_team in combinations(agents, first_team_size):
            first_team = list(first_team)
            remaining_agents = [a for a in agents if a not in first_team]
            sub_partitions = self._generate_team_partitions(remaining_agents, remaining_sizes)

            for sub_partition in sub_partitions:
                partitions.append([first_team] + sub_partition)

        return partitions

    def expert_policy(self, obs, goal_type, agent_idx):
        """Use trained PPO policy for the given goal."""
        # Map goal_type to goal_name used in training
        goal_name_map = {
            'top_right': 'team_goal_top_right',
            'top_left': 'team_goal_top_left',
            'bottom_left': 'team_goal_bottom_left',
            'bottom_right': 'team_goal_bottom_right'
        }

        goal_name = goal_name_map.get(goal_type)
        if not goal_name:
            # Fallback to simple policy if goal not recognized
            return self.simple_policy(obs, goal_type, agent_idx)

        # Find the corresponding agent/model for this goal
        try:
            goal_idx = self.goal_names.index(goal_name)
            agent = self.agents[goal_idx]

            # Extract this agent's position from the full observation
            pos_offset = agent_idx * 2
            agent_pos = obs[pos_offset:pos_offset+2]

            # Get action probabilities from the trained model
            action_probs = agent.get_action_probabilities(agent_pos)

            # Sample action from the distribution (or take argmax for deterministic)
            # For visualization, we'll use deterministic (argmax) for consistency
            action = int(np.argmax(action_probs))

            return action
        except (ValueError, IndexError, AttributeError) as e:
            # If there's any error, fall back to simple policy
            print(f"Warning: Failed to use trained policy for {goal_type}, using simple policy. Error: {e}")
            return self.simple_policy(obs, goal_type, agent_idx)

    def simple_policy(self, obs, goal_type, agent_idx):
        """Simple fallback policy that moves towards goal (doesn't handle obstacles well)."""
        goal_positions = {
            'top_right': np.array([6, 0]),
            'top_left': np.array([0, 0]),
            'bottom_left': np.array([0, 6]),
            'bottom_right': np.array([6, 6])
        }

        goal_pos = goal_positions[goal_type]
        pos_offset = agent_idx * 2
        agent_pos = obs[pos_offset:pos_offset+2]

        diff = goal_pos - agent_pos
        if abs(diff[0]) > abs(diff[1]):
            return 3 if diff[0] > 0 else 2  # Right or Left
        else:
            return 0 if diff[1] < 0 else 1  # Up or Down


class ConfigDialog(QDialog):
    """Dialog for selecting configuration options."""

    def __init__(self, available_episodes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

        # Set dialog background and text colors
        self.setStyleSheet("""
            QDialog {
                background-color: #ecf0f1;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Select Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
                color: #2c3e50;
                background-color: white;
                border-radius: 5px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(15)

        # Episodes selector label
        episodes_label = QLabel("Training Episodes:")
        episodes_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        self.episodes_combo = QComboBox()
        for ep in available_episodes:
            self.episodes_combo.addItem(f"{ep} episodes", ep)
        self.episodes_combo.setStyleSheet("""
            QComboBox {
                padding: 10px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
                color: #2c3e50;
                font-size: 14px;
                min-width: 200px;
                min-height: 35px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #2c3e50;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #2c3e50;
                selection-background-color: #3498db;
                selection-color: white;
            }
        """)
        form.addRow(episodes_label, self.episodes_combo)

        # Scenario selector label
        scenario_label = QLabel("Scenario:")
        scenario_label.setStyleSheet(episodes_label.styleSheet())

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("2 Teams (1 agent each)", (1, 1))
        self.scenario_combo.addItem("2 Teams (2 agents each)", (2, 2))
        self.scenario_combo.setStyleSheet(self.episodes_combo.styleSheet())
        form.addRow(scenario_label, self.scenario_combo)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: 2px solid #2980b9;
                padding: 10px 20px;
                border-radius: 5px;
                min-width: 100px;
                min-height: 40px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_config(self):
        """Get selected configuration."""
        episodes = self.episodes_combo.currentData()
        team_sizes = self.scenario_combo.currentData()
        return episodes, list(team_sizes)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Agent Goal Recognition GUI Visualizer')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes used for training (if not specified, will prompt)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory with trained models')
    args = parser.parse_args()

    print("="*80)
    print(" " * 20 + "MODERN GUI VISUALIZER")
    print("="*80)

    # Configuration
    MODELS_DIR = args.models_dir

    # Scan for available trained models
    print("\nScanning for trained models...")
    available_episodes = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            if item.startswith('episodes_'):
                try:
                    ep_count = int(item.split('_')[1])
                    available_episodes.append(ep_count)
                except (IndexError, ValueError):
                    pass

    available_episodes.sort(reverse=True)

    if not available_episodes:
        print(f"\n❌ Error: No trained models found in {MODELS_DIR}/")
        print("Please train models first using: python train.py --episodes <N>")
        return

    print(f"Found models for episodes: {available_episodes}")

    # Start Qt application early for dialog
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style

    # Get configuration
    if args.episodes is not None:
        # Use command-line argument
        if args.episodes not in available_episodes:
            print(f"\n❌ Error: Models for {args.episodes} episodes not found!")
            print(f"Available: {available_episodes}")
            return
        EPISODES = args.episodes
        team_sizes = [1, 1]  # Default scenario
    else:
        # Show configuration dialog
        config_dialog = ConfigDialog(available_episodes)
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            EPISODES, team_sizes = config_dialog.get_config()
        else:
            print("\nCancelled by user.")
            return

    goal_names = [
        "team_goal_top_right",
        "team_goal_top_left",
        "team_goal_bottom_left",
        "team_goal_bottom_right"
    ]

    # Verify models exist for selected episodes
    models_exist = all(
        os.path.exists(os.path.join(
            MODELS_DIR, f"episodes_{EPISODES}", goal, "PPOAgent", "model.pt"
        ))
        for goal in goal_names
    )

    if not models_exist:
        print(f"\n❌ Error: Some models missing for episodes_{EPISODES}!")
        print(f"Please run 'python train.py --episodes {EPISODES}' first.")
        return

    # Load trained agents
    print(f"\nLoading trained agents (episodes={EPISODES})...")
    env_classes = [
        TeamGoalTopRight,
        TeamGoalTopLeft,
        TeamGoalBottomLeft,
        TeamGoalBottomRight
    ]

    agents = []
    for env_class, goal_name in zip(env_classes, goal_names):
        agent = PPOAgent(
            env_name=env_class,
            models_dir=MODELS_DIR,
            goal_hypothesis=goal_name,
            episodes=EPISODES
        )
        agent.load_model()
        agents.append(agent)
        print(f"  ✓ Loaded {goal_name}")

    # Configuration for scenario
    team_goals = ['top_right', 'top_left']
    num_agents = sum(team_sizes)

    print("\nStarting GUI...")
    print(f"Episodes: {EPISODES}")
    print(f"Team configuration: {team_sizes}")
    print(f"Team goals: {team_goals}")

    window = VisualizerMainWindow(agents, goal_names, num_agents, team_sizes, team_goals)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
