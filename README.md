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

This trains 4 policies (one per team goal):
- `team_goal_top_right`
- `team_goal_top_left`
- `team_goal_bottom_left`
- `team_goal_bottom_right`

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

### StarCraft Scenario Workflow

- Maps live under `starcraft-maps/sc1-map/` and their associated start/goal
  scenarios reside in `starcraft-maps/sc1-scen/`.
- Train PPO policies for the first N scenarios using
  ```bash
  python train_starcraft.py --map-id Aftershock --episodes 5000 --scenario-count 10
  ```
  which writes one checkpoint per scenario inside
  `models/starcraft/<map_id>/episodes_<episodes>/` and records the automatically
  derived horizon/penalty/shape settings. Use `--scenario-index` /
  `--scenario-id` to target specific lines, or `--max-steps-scale` to override
  the auto scaling.
  Use `--scenario-index` or `--scenario-id` to target a specific line.
- Evaluate trained policies (for reuse in recognition) via
  ```bash
  python recognize_starcraft.py --map-id Aftershock --train-episodes 5000 --rollouts 10
  ```
  to report success rates and reward statistics. See
  `docs/starcraft_workflow.md` for details on extending or filtering scenarios.
- Visualize a trained scenario directly on the high-resolution PNG with
  ```bash
  python visualize_starcraft.py --map-id Aftershock --episodes 5000 --scenario-index 0
  ```
  and watch the greedy policy trace over `sc1-png/<map>.png`. Use `+` / `-`
  (or the mouse wheel) to zoom, drag to pan, and press `0` to reset view.

## Multi-Agent Environment

### Grid World Setup
- **Size**: 7×7 grid
- **Teams**: Configurable (e.g., 2 teams)
- **Team sizes**: Configurable (e.g., [1, 1] or [2, 2])
- **Goals**: Each team assigned a distinct corner objective
- **Obstacles**: Static cross-shaped barriers with offset blocks that force detours
- **Initial positions**: Canonical spawn presets (e.g., agents targeting top-right/top-left start at (1, 6) and (5, 6)) ensure reproducible comparisons; override via `initial_agent_positions` or switch presets with `start_preset` for custom scenarios
- **Observability**: Full (all agents see all positions)

The presets exposed through `envs.DEFAULT_INITIAL_POSITION_PRESETS` cover common two-team layouts; add more (or pass explicit `initial_agent_positions` / adjust `start_preset`) to investigate additional initial-state/goal pairs.

### Team Goals
1. **top_right**: Reach position (6, 0)
2. **top_left**: Reach position (0, 0)
3. **bottom_left**: Reach position (0, 6)
4. **bottom_right**: Reach position (6, 6)

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

### Latency Reporting

In addition to accuracy, the CLI demo and recognition scripts now report how many observations were required to lock onto the correct goals and team assignments (goal/team/joint lock-in). Use these ratios (e.g., `goal lock-in: 12/30`) to judge convergence speed under the obstacle maze.

### Command Cheat Sheet

- `pip install -r requirements.txt` – install the Python dependencies (PyTorch, Gymnasium, NumPy, PyQt6) into your active environment.
- `python train.py --episodes 5000 [--device cuda]` – train the four corner-reaching policies. Training still samples random non-obstacle starting cells so the policies generalise across corridors.
- `python recognize.py --episodes 5000 --scenario 0 [--metric kl_divergence]` – run all recognition scenarios sequentially using the deterministic multi-agent spawn presets (Scenario 0 ≡ Scenarios 1–3 back-to-back). Individual scenarios are: 1 = two teams / one agent per team, 2 = two teams / two agents each, 3 = same as 1 with noisy policies.
- `python demo.py --episodes 5000 --mode full` – console “story mode” that (re)trains if checkpoints are missing, shows the visual step-through, and finishes with the recognition summary. Use `--mode visualize` or `--mode recognize` to run a single phase.
- `python incremental_recognition.py` – interactive CLI that replays a scenario step-by-step, printing obstacles, spawn presets, observations, and latency after each action.
- `python visualizer_gui.py --episodes 5000` – launch the PyQt GUI with the obstacle-aware board, deterministic spawn preset, confidence bars, and latency panel. You can select a different preset or team sizing from the configuration dialog when prompted.

Tip: to experiment with additional deterministic start states, either extend `envs.DEFAULT_INITIAL_POSITION_PRESETS` or instantiate `MultiAgentGridWorld(..., initial_agent_positions=[])` in your own driver script.

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
