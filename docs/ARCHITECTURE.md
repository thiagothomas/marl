# System Architecture and Workflow

## 1) Overview
This project performs multi-agent goal and team recognition by composing single-agent policies trained for specific team goals (the four grid corners). The pipeline runs in two phases:
- Phase 1 (Training): Train one PPO policy per goal in single‑agent, obstacle‑aware grids.
- Phase 2 (Recognition): Observe multiple agents, score all team/goal hypotheses using the trained policies, and pick the best assignment. Results include accuracy and latency (observations to lock‑in).

Key features:
- Static cross‑shaped obstacles that force detours (shared during training and evaluation).
- Deterministic spawn presets in the multi‑agent arena for reproducible comparisons.
- Multiple evaluation metrics (KL divergence, cross‑entropy, mean action distance).
- Console demos and a PyQt GUI with confidence and latency panels.

## 2) Phase 1 — Single‑Agent Policy Training
- Environments: `envs/team_goal_environments.py` declares four single‑agent grids, each inheriting `GridWorldBase` (`envs/grid_world.py`). The base adds the obstacle layout and disallows spawning/moving into blocked cells.
  - Goals: `top_right`, `top_left`, `bottom_left`, `bottom_right`.
  - Reward: +10 on goal; small negative proportional to Euclidean distance otherwise.
  - Starts: randomized non‑goal, non‑obstacle cells (policies learn to traverse corridors).
- Algorithm: Proximal Policy Optimization (PPO) in `ml/ppo.py`. A separate agent is trained for each goal, producing checkpoints under `models/episodes_<N>/team_goal_*/PPOAgent/`.
- Entrypoint: `python train.py --episodes <N> [--device cuda]`.
  - Use the same `<N>` later for recognition/demos so checkpoints line up.

Why single‑agent? It simplifies learning and produces stable goal‑conditioned policies. During recognition we “compose” these policies to explain multi‑agent behavior.

## 3) Phase 2 — Multi‑Agent Recognition
- Environment: `envs/multi_agent_grid_world.py` hosts the arena and convenience classes:
  - `TwoTeamsSingleAgent` (1 agent/team), `TwoTeamsDoubleAgents` (2 agents/team).
  - Obstacles: same cross pattern as training.
  - Deterministic spawns: `DEFAULT_INITIAL_POSITION_PRESETS` hard‑codes initial coordinates for reproducibility; override with `initial_agent_positions` or switch via `start_preset`.
- Observation collection: Given hidden “expert” or noisy policies, we record per‑agent (observation, action) pairs for a fixed horizon.
- Hypothesis space:
  1) Generate all team partitions matching required team sizes.
  2) Combine with corner‑goal assignments for each team.
- Scoring: For a hypothesis, score each agent’s trajectory against the policy trained for the assigned team goal. Higher is better.
  - `kl_divergence` (default): negative KL between observed actions and the policy.
  - `cross_entropy`: negative log‑likelihood under the policy.
  - `mean_action_distance`: simple action mismatch proxy.
- Search: Brute‑force over partitions × goal combos; select best total score.
- Latency: Recompute best assignment using only the first k observations (k=1…T) and report earliest k where goals, teams, or the full joint assignment are correct.
- Entrypoints:
  - `python recognize.py --episodes <N> --scenario 0` (run all scenarios 1–3).
  - `python incremental_recognition.py` (step‑by‑step CLI visualizer).
  - `python visualizer_gui.py --episodes <N>` (PyQt GUI with obstacles, confidence, latency).

## 4) Scenarios (recognize.py)
- Scenario 1: Two teams, one agent each. Team A → top_right; Team B → top_left. Env: `TwoTeamsSingleAgent` with a fixed spawn preset.
- Scenario 2: Two teams, two agents each. Same goals; more agents. Env: `TwoTeamsDoubleAgents` with a fixed preset.
- Scenario 3: Same as Scenario 1, but observations are generated from noisy (mixed) policies to test robustness.
- Scenario 0: Runs 1→3 sequentially and prints a summary.

## 5) Deterministic Presets and Reproducibility
- Presets live in `DEFAULT_INITIAL_POSITION_PRESETS` and are printed at runtime.
- Use them to compare algorithms under identical initial state/goal pairs.
- Add presets or pass `initial_agent_positions` to evaluate specific cases.

## 6) Visualization & Diagnostics
- CLI: `incremental_recognition.py` renders the grid in text, prints obstacles, observations, and top‑k hypotheses each step.
- GUI: `visualizer_gui.py` draws the grid, obstacles, agent paths, per‑agent goal confidence bars, top‑k hypotheses, and lock‑in latency.
- Tools print selected preset and obstacle info for transparency.

## 7) Extending the System
- New scenarios: extend `DEFAULT_INITIAL_POSITION_PRESETS` or provide `initial_agent_positions` directly when constructing the env.
- New metrics: implement a function in `metrics/metrics.py` and hook it into `recognize.py --metric`.
- New goals: add a `TeamGoal*` class, register it in `train.py`/`recognize.py`, retrain, and include it in goal combinations. (Current focus is the four corners.)

## 8) Practical Notes & Limitations
- Complexity: brute‑force search scales combinatorially with agent count and team partitions; add pruning if needed.
- Matching episodes: keep `--episodes` consistent between training and recognition so checkpoints load.
- Robustness: single‑agent random starts encourage corridor generalization; evaluation uses fixed starts for clean comparisons.
- Determinism: recognition is deterministic given preset + policies; training remains stochastic (set seeds if required).

## 9) Quick Start
1) Install deps: `pip install -r requirements.txt`.
2) Train: `python train.py --episodes 5000 --device cuda` (or `cpu`).
3) Recognize: `python recognize.py --episodes 5000 --scenario 0`.
4) Visualize: `python visualizer_gui.py --episodes 5000` or `python incremental_recognition.py`.
