# Command Reference

This guide lists the primary workflow commands, explains why and when to run them, and documents key parameters so you can adapt runs quickly.

## Environment Setup

- `python -m venv .venv && source .venv/bin/activate`  
  *When*: before any Python work in a fresh clone.  
  *What it does*: creates and activates an isolated virtual environment to keep dependencies separate.  
- `pip install -r requirements.txt`  
  *When*: after activating the virtualenv or when requirements change.  
  *What it does*: installs core runtime libraries like Gymnasium, PyTorch, and NumPy.  
  *Notable flags*: add `--extra-index-url` or `--upgrade` as needed for custom mirrors or newer packages.

## Grid-World Training & Recognition

- `python train.py --episodes 5000 --models-dir models/`  
  *Run when*: you change policy logic (`ml/`) or want fresh checkpoints.  
  *Effect*: trains PPO agents for each corner goal, storing them under `models/episodes_5000/`.  
  *Key parameters*:  
  - `--episodes`: training horizon per goal; higher values improve convergence but take longer.  
  - `--models-dir`: redirect outputs (e.g., `models/experiments/`).  
  - `--device`: `cpu` or `cuda`; defaults to auto-detection.  
- `python recognize.py --episodes 5000 --metric kl_divergence --scenario 0`  
  *Run when*: evaluating recognition quality after training updates.  
  *Effect*: loads saved policies, simulates multi-agent scenarios, and scores plans using KL divergence.  
  *Key parameters*:  
  - `--episodes`: must match the training run you want to load.  
  - `--metric`: switch to `cross_entropy` or `mean_action_distance` for different scoring heuristics.  
  - `--scenario`: choose `0` for the full suite, or 1–3 for targeted tests.  
  - `--observation-steps`: number of steps per observation sequence (defaults to 30).  
- `python incremental_recognition.py --observation-steps 30`  
  *Run when*: debugging recognition step-by-step or gathering visual evidence.  
  *Effect*: launches an interactive CLI that prints evolving belief states for each hypothesis.  
  *Key parameters*: tweak `--episodes` to match the checkpoint set, or `--observation-steps` to mimic different sensor windows.

## StarCraft Scenario Training & Evaluation

- `python train_starcraft.py --map-id Aftershock --episodes 5000 --scenario-count 10 --device cpu`  
  *Run when*: you want PPO policies for Moving AI StarCraft maps (`starcraft-maps/sc1-map`).  
  *Effect*: parses the `.map.scen` scenarios for the map, trains one agent per line, and saves checkpoints in `models/starcraft/<map>/episodes_<episodes>/`.  
  *Key parameters*:  
  - `--map-id`: must match the file prefix (e.g., `Aftershock` ⇒ `Aftershock.map` / `Aftershock.map.scen`).  
  - `--scenario-count`: limit the number of scenarios (set to `0` to train all).  
  - `--scenario-index` / `--scenario-id`: train a specific scenario line instead of a range.  
  - `--max-steps-scale`: override the automatically derived horizon (defaults use scenario bucket and optimal distance). Progress-shaping rewards are scaled automatically from the same metadata.  
  - Rollout length scales automatically with scenario horizon (128–4096 steps) so each PPO update sees several full episodes.  
  - `--entropy-coef`: initial exploration bonus (decays toward zero automatically during training).  
  - Agents default to 8-directional movement with a 3×3 occupancy patch; override via the StarCraft config if needed.  
  - `--device`: switch to `cuda` when GPUs are available.  
- `python recognize_starcraft.py --map-id Aftershock --train-episodes 5000 --rollouts 10`  
  *Run when*: validating trained StarCraft policies before using them in recognition workflows.  
  *Effect*: reloads checkpoints, runs greedy rollouts per scenario, and prints success rates plus reward stats.  
  *Key parameters*:  
  - `--train-episodes`: must match the directory created during training.  
  - `--rollouts`: number of evaluation episodes per scenario; increase for more stable averages.  
  - `--scenario-count`: mirror the training subset if you only trained a portion of the file.  
  - `--max-steps-scale`: optional override; otherwise the viewer uses the runtime settings saved during training.  
- `python visualize_starcraft.py --map-id Aftershock --episodes 5000 --scenario-index 0`  
  *Run when*: you want a visual replay of a trained policy moving across the original PNG map.  
  *Effect*: loads the checkpoint, overlays the trajectory on `sc1-png/<map>.png`, and animates steps in a PyQt window.  
  *Key parameters*:  
  - `--scenario-index` / `--scenario-id`: select which start/goal pair to replay.  
  - `--step-interval`: animation speed in milliseconds (lower = faster).  
  - `--max-steps-scale`: optional override if you experimented with custom horizons.  
  - `--device`: switch to `cuda` if the model was saved with GPU tensors.  
  *Viewer controls*: use `+` / `-` or the mouse wheel to zoom, drag to pan, and press `0` to reset.

## Interactive Testing & Visualization

- `python demo.py --episodes 5000 --models-dir models/`  
  *Run when*: sanity-checking policies in the grid world with live visualization.  
  *Effect*: loads trained corner-goal agents and simulates them navigating the obstacle field.  
  *Key parameters*:  
  - `--episodes`: choose the checkpoint set (defaults to 5000).  
  - `--team-sizes`: customize team compositions (e.g., `--team-sizes 2 1`).  
- `python incremental_recognition.py --episodes 5000 --observation-steps 40`  
  *Run when*: exploring how changing observation windows alters hypothesis ranking.  
  *Effect*: similar to the earlier entry but emphasized here for experimentation; use slightly longer observation windows when agents need extra moves around new obstacles.

## Maintenance & Utilities

- `python visualizer_gui.py`  
  *Run when*: you need a GUI to inspect trajectories or to present demos.  
  *Effect*: opens a PyQt6-based viewer hooked into the trained policies. Close and reopen after training new checkpoints to refresh loads.  
- `python delete_non_512x512_maps.py --root starcraft-maps`  
  *Run when*: pruning StarCraft map folders to only include high-resolution maps.  
  *Effect*: removes maps that are not 512×512 from `sc1-map/`, `sc1-png/`, and `sc1-scen/`. Take care—this is destructive.  

## Tips for Parameter Sweeps

- Use shell loops or `xargs` to iterate over multiple maps or scenario counts:  
  ```bash
  for map in Aftershock Brushfire; do
    python train_starcraft.py --map-id "$map" --episodes 3000 --scenario-count 5
  done
  ```  
- Capture logs with `tee` to keep a record of rewards and success rates:  
  ```bash
  python recognize.py --episodes 5000 --metric kl_divergence | tee logs/recognize_$(date +%F).log
  ```  
- When training on GPU, pin the device once via env var:  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python train_starcraft.py --map-id Aftershock --device cuda
  ```  

Keep this reference handy when kicking off new experiments or onboarding collaborators so everyone runs the same command set with full context.
