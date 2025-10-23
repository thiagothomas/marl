# StarCraft Workflow

The StarCraft pipeline uses Moving AI map (`.map`) and scenario (`.map.scen`)
files to produce PPO policies that mirror the grid-world training workflow.

## 1. Scenario Selection

The repository already includes maps under `starcraft-maps/sc1-map/` and the
matching scenario files under `starcraft-maps/sc1-scen/`. Each line in a
`.map.scen` file defines a start/goal pair. By default the CLI scripts use the
first ten scenarios, but you can switch to a different subset via
`--scenario-count` or edit the `.scen` file to reorder entries.

## 2. Training PPO Policies

Run `train_starcraft.py` to train one PPO agent per scenario:

```bash
python train_starcraft.py --map-id Aftershock --episodes 5000 --scenario-count 10
```

Use `--entropy-coef` (default `1e-3`) to adjust the initial PPO entropy bonus; the trainer now
linearly anneals it toward zero so policies sharpen automatically. Rollout length is chosen from the
scenario horizon (128–4096) so each PPO update captures several full episodes. Progress shaping
defaults to `0.2`, making backtracking near goals carry a steeper penalty. Agents observe a 3×3
occupancy window and have 8-directional movement enabled by default.

Checkpoints are stored under `models/starcraft/<map_id>/episodes_<episodes>/`.
Each scenario directory contains `model.pt` (the PPO weights) and `config.json`
with the start/goal metadata plus the derived runtime parameters (max steps,
penalties, progress-shaping scale, reward). Use `--scenario-index` or
`--scenario-id` when you only need
a specific line from the `.map.scen` file; horizons scale automatically based on
scenario difficulty and optimal path length but can be overridden via
`--max-steps-scale`.

## 3. Evaluating for Recognition

Use `recognize_starcraft.py` to reload the saved models and estimate success
rates via greedy rollouts. Runtime settings saved during training are reused by
default:

```bash
python recognize_starcraft.py --map-id Aftershock --train-episodes 5000 --rollouts 10
```

The script reports success percentages and reward statistics per scenario. The
trained models can then be imported into downstream recognition pipelines that
expect PPO agents saved with `ml/ppo.py`.

## 4. Visual Playback

Use the PyQt viewer to play a trained policy over the original PNG asset:

```bash
python visualize_starcraft.py --map-id Aftershock --episodes 5000 --scenario-index 0
```

Adjust `--scenario-index` (or `--scenario-id`) to switch start/goal pairs, and
tune `--step-interval` for faster or slower animation. Use the `+` / `-` keys or
the mouse wheel to zoom, drag to pan, and press `0` to reset. The overlay highlights start
(green), goal (blue), the traversed path, and the current agent position.
