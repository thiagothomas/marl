"""Multi-agent environments for team-based goal recognition."""

from .grid_world import GridWorldBase

from .multi_agent_grid_world import (
    MultiAgentGridWorld,
    TwoTeamsSingleAgent,
    TwoTeamsDoubleAgents
)

from .team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight,
    TeamGoalCenter
)

__all__ = [
    'GridWorldBase',
    'MultiAgentGridWorld',
    'TwoTeamsSingleAgent',
    'TwoTeamsDoubleAgents',
    'TeamGoalTopRight',
    'TeamGoalTopLeft',
    'TeamGoalBottomLeft',
    'TeamGoalBottomRight',
    'TeamGoalCenter'
]