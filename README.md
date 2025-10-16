# Multi-Agent Team-Based Goal Recognition

Multi-agent goal recognition system using reinforcement learning. The system trains PPO agents for different team goals and uses these policies to recognize both **team membership** and **team goals** from observed multi-agent behaviors.

## Overview

This extends the DRACO approach to multi-agent scenarios where multiple teams of agents pursue different objectives simultaneously.

### Problem Statement
Given observations of multiple agents acting in an environment:
- Which agents belong to which team?
- What goal is each team pursuing?

### Two-Phase Pipeline

**Phase 1: Training Team Goal Policies**
- Train separate PPO agents for each possible team goal
- Each agent learns an optimal policy for reaching its specific goal
- Training uses single-agent environments for simplicity
- Trained models are saved for later use

**Phase 2: Multi-Agent Recognition**
- Observe multiple agents acting simultaneously
- Generate all possible team partitions and goal assignments
- Score each hypothesis by matching observed behaviors to trained policies
- Return ranked assignments with confidence scores

## Project Structure

```
.
├── envs/                       # Grid world environments
│   ├── grid_world.py          # Base environment class
│   ├── team_goal_environments.py  # Single-agent training envs
│   └── multi_agent_grid_world.py  # Multi-agent testing env
├── ml/                         # Machine learning components
│   ├── base_agent.py          # Abstract RL agent
│   ├── ac_model.py            # Actor-Critic network
│   └── ppo.py                 # PPO implementation
├── metrics/                    # Evaluation metrics
│   └── metrics.py             # KL divergence, cross-entropy, etc.
├── recognizer/                 # Goal recognition logic
│   ├── recognizer.py          # Base recognizer
│   └── multi_agent_recognizer.py  # Multi-agent recognizer
├── models/                     # Trained models
├── train.py                    # Train team goal policies
├── recognize.py                # Run recognition experiments
├── demo.py                     # Interactive demo
├── incremental_recognition.py  # Step-by-step visualizer
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- Gymnasium 0.29+
- NumPy 1.24+

## Usage

### Quick Start: Interactive Demo

```bash
python demo.py --episodes 5000
```

This will train models (if needed) and let you interactively test multi-agent recognition.

### Step-by-Step Workflow

#### 1. Train Team Goal Policies

```bash
python train.py --episodes 5000
```

This trains 5 policies (one per team goal):
- `team_goal_top_right`
- `team_goal_top_left`
- `team_goal_bottom_left`
- `team_goal_bottom_right`
- `team_goal_center`

Options:
- `--episodes`: Training episodes per agent (default: 5000)
- `--models-dir`: Model save directory (default: `models`)
- `--device`: Use `cpu` or `cuda` (default: `cpu`)

#### 2. Run Recognition Experiments

```bash
python recognize.py --episodes 5000
```

Tests recognition on multiple scenarios:
- 2 teams, 1 agent each
- 2 teams, 2 agents each
- Various team goal combinations

Options:
- `--episodes`: Must match training (default: 5000)
- `--observation-steps`: Steps to observe (default: 30)
- `--metric`: `kl_divergence`, `cross_entropy`, or `mean_action_distance`
- `--scenario`: Run specific scenario (0=all, 1-3=specific)

#### 3. Step-by-Step Visualization

```bash
python incremental_recognition.py
```

Interactive visualizer that shows:
- Multi-agent grid world with team movements
- Incremental belief updates as observations accumulate
- Top-k hypothesis rankings at each step
- Final team assignment prediction

## Multi-Agent Environment

### Grid World Setup
- **Size**: 7×7 grid
- **Teams**: Configurable (e.g., 2 teams)
- **Team sizes**: Configurable (e.g., [1, 1] or [2, 2])
- **Goals**: Each team assigned a corner or center position
- **Observability**: Full (all agents see all positions)

### Team Goals
1. **top_right**: Reach position (6, 0)
2. **top_left**: Reach position (0, 0)
3. **bottom_left**: Reach position (0, 6)
4. **bottom_right**: Reach position (6, 6)
5. **center**: Reach position (3, 3)

### Observation Space
Each agent observes a flat array of all agent positions:
```python
obs = [agent_0_x, agent_0_y, agent_1_x, agent_1_y, ..., agent_N_x, agent_N_y]
```

### Action Space
4 discrete actions per agent:
- 0: Up (y - 1)
- 1: Down (y + 1)
- 2: Left (x - 1)
- 3: Right (x + 1)

### Rewards
- **+10.0**: First team member reaches goal
- **-0.01 × distance**: Distance penalty while approaching
- **0.0**: After team succeeds

## How Recognition Works

### Team Assignment as Combinatorial Search

For N agents and M teams with specified sizes:

1. **Generate team partitions**: All ways to split N agents into M groups
   - Example: 2 agents → 2 teams of size 1 → [[0], [1]] and [[1], [0]]

2. **For each partition + goal combination**:
   - Match each agent's behavior to the assigned team goal policy
   - Compute score (e.g., negative KL divergence)
   - Sum scores across all agents

3. **Rank hypotheses**: Best assignment has highest total score

### Evaluation Metrics

- **KL Divergence** (primary): Measures policy distribution difference
- **Cross-Entropy**: Negative log-likelihood of observed actions
- **Mean Action Distance**: Simple action matching

### Example Recognition Output

```
Top Team Assignment Hypotheses:

✓ #1   Score:   -3.97
      Team 0: Agents ['0'] → team_goal_top_right
      Team 1: Agents ['1'] → team_goal_top_left
      Breakdown: agent_0→team_goal_top_right: -0.46, agent_1→team_goal_top_left: -3.51
      *** TRUE ASSIGNMENT ***

  #2   Score:   -3.97
      Team 0: Agents ['1'] → team_goal_top_left
      Team 1: Agents ['0'] → team_goal_top_right
      Breakdown: agent_1→team_goal_top_left: -3.51, agent_0→team_goal_top_right: -0.46
```

Note: Identical scores indicate symmetric assignments (team labels are arbitrary).

## Extending the System

### Adding New Team Goals

1. Create environment in `envs/team_goal_environments.py`:
```python
class TeamGoalCustom(GridWorldBase):
    def _set_goal_position(self):
        self.goal_pos = np.array([x, y], dtype=np.int32)

    def _calculate_reward(self):
        if self._is_goal_reached():
            return 10.0
        distance = np.linalg.norm(
            self.agent_pos.astype(float) - self.goal_pos.astype(float)
        )
        return -0.01 * distance
```

2. Update `train.py`:
```python
team_goal_classes = [..., TeamGoalCustom]
goal_names = [..., "team_goal_custom"]
```

3. Retrain all policies

## Performance Tips

### Training
- **Quick testing**: 1,000-5,000 episodes
- **Good performance**: 10,000-50,000 episodes
- **Best results**: 100,000+ episodes
- Use `--device cuda` for GPU acceleration

### Recognition
- **Minimum observations**: 10-20 steps
- **Recommended**: 30-50 steps
- **Complex scenarios**: 100+ steps
- KL divergence usually performs best

## Troubleshooting

### Low Recognition Accuracy
- Increase training episodes (10,000+)
- Collect more observation steps (50+)
- Verify agents are behaving optimally in their respective environments

### Model Loading Errors
- Ensure episode count matches: `train.py --episodes 5000` → `recognize.py --episodes 5000`
- Check models directory exists: `ls models/episodes_5000/`

### Symmetric Score Ties
- Expected behavior with symmetric team sizes
- Team labels are arbitrary - focus on agent-goal matching
- Consider tie-breaking strategies if needed

## Implementation Notes

### Key Design Decisions

1. **Independent training**: Each team goal gets its own single-agent training environment
2. **Brute-force search**: Recognition tries all possible team assignments
3. **Full observability**: Agents see all positions (simplifies recognition)
4. **Team-level termination**: All team members stop when any reaches goal

### Future Extensions

- Partial observability
- Communication between team members
- Coordinated multi-agent policies (MARL)
- Online/incremental learning

## References

Based on:
- **DRACO**: Goal Recognition using Deep Reinforcement Learning
- **GRAQL**: Goal Recognition using Q-Learning

Extended to multi-agent team-based scenarios with combinatorial assignment search.
