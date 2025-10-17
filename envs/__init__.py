"""Multi-agent environments for team-based goal recognition."""

from .grid_world import GridWorldBase

from .multi_agent_grid_world import (
    MultiAgentGridWorld,
    TwoTeamsSingleAgent,
    TwoTeamsDoubleAgents,
    DEFAULT_INITIAL_POSITION_PRESETS
)

from .team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight
)

__all__ = [
    'GridWorldBase',
    'MultiAgentGridWorld',
    'TwoTeamsSingleAgent',
    'TwoTeamsDoubleAgents',
    'DEFAULT_INITIAL_POSITION_PRESETS',
    'TeamGoalTopRight',
    'TeamGoalTopLeft',
    'TeamGoalBottomLeft',
    'TeamGoalBottomRight'
]
