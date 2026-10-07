# Sample-Efficiency of Kolmogorov-Arnold Networks

Deep reinforcement learning has achieved substantial performance gains over classical control approaches.
Yet, a central challenge to learning in real-world applications is acquiring costly samples.
Kolmogorov-Arnold Networks are a recently proposed architecture that can learn physical relationships in control problems effectively, with significantly higher parameter efficiency and interpretability when compared to Multi-Layer-Perceptron architectures.
In this work, we systematically study sample-efficiency using computational experiments, covering the Feynman dataset and the Gymnasium RL benchmark.
The results show that similar performance can be achieved with 40\% fewer samples using the Kolmogorov-Arnold architecture, and that relative performance improvements up to 50\% occur during the training process.
The observed gains are robust to varying levels of noise in rewards.
These results highlight the potential of the Kolmogorov-Arnold architectures for more sample-efficient reinforcement learning.

**Authors**: Kevin Riehl, Shaimaa K. El-Baklish, Fan Wu, Anastasios Kouvelas

**Institution**: ETH Zürich (2026)

## Paper at a Glance

The manuscript studies KANs in three stages:

1. **Feynman sample-efficiency benchmark** - supervised regression on analytic physics equations from the AI-Feynman/Feynman benchmark family.
2. **Gymnasium Classic Control benchmark** - PPO agents with KAN and MLP actor architectures on low-dimensional control environments.
3. **Noisy-reward Gymnasium benchmark** - the same PPO setting under increasing Gaussian reward noise.

The paper reports that KANs can achieve comparable performance with substantially fewer samples in the studied low-dimensional, physics-structured settings. The strongest gains appear in the early and middle training regimes: roughly 40% fewer samples for comparable performance in the reported curves, with reward improvements reaching about 50% during parts of training. At long training budgets, KANs and MLPs often converge toward similar final performance.

## Code Structure

### `src/experiment_1_feynman`

This experiment isolates the approximation problem before moving to RL. It samples analytic Feynman equations, trains KAN and MLP regressors at different training-set sizes, and evaluates held-out regression error.

Important files:

- [`feynman_db.py`](src/experiment_1_feynman/feynman_db.py): vectorized registry of benchmark equations and sampling domains.
- [`experiment_protocol_main.py`](src/experiment_1_feynman/experiment_protocol_main.py): cluster-style sweep definition, including sample sizes, seeds, model families, and parameter-matched MLP setup selection.
- [`experiment_runner.py`](src/experiment_1_feynman/experiment_runner.py): CSV-producing runner with checkpointing and resumable result handling.
- [`model_kan.py`](src/experiment_1_feynman/model_kan.py): wrapper around selected models from the vendored All-KAN code.
- [`model_mlp.py`](src/experiment_1_feynman/model_mlp.py): comparable MLP baseline.
- [`visualization_results.py`](src/experiment_1_feynman/visualization_results.py): analysis script for aggregating raw CSVs into plots and summary tables.

### `src/experiment_2_gymnasium`

This experiment compares KAN and MLP PPO agents on Gymnasium Classic Control tasks:

- `Acrobot-v1`
- `CartPole-v1`
- `MountainCar-v0`
- `MountainCarContinuous-v0`
- `Pendulum-v1`

The protocol sweeps KAN B-spline and KAN Gaussian RBF actors against ReLU and sigmoid MLP actors. The critic is kept as an MLP by default, so the comparison focuses primarily on actor architecture.

Important files:

- [`experiment_classic_protocol_main.py`](src/experiment_2_gymnasium/experiment_classic_protocol_main.py): environment list, model sweep, PPO hyperparameters, and parallel-run settings.
- [`experiment_classic_runner.py`](src/experiment_2_gymnasium/experiment_classic_runner.py): PPO implementation, evaluation loop, checkpoint CSV writing, and resume logic.
- [`demoscript_cartpole.py`](src/experiment_2_gymnasium/demoscript_cartpole.py): smaller paired KAN/MLP CartPole demo.

### `src/experiment_3_gymnasium_noise`

This extends the Gymnasium PPO benchmark with Gaussian reward noise. The protocol uses the same Classic Control environment family and tests reward-noise levels:

```text
0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0
```

Important files mirror experiment 2:

- [`experiment_classic_protocol_main.py`](src/experiment_3_gymnasium_noise/experiment_classic_protocol_main.py): noisy-reward sweep definition.
- [`experiment_classic_runner.py`](src/experiment_3_gymnasium_noise/experiment_classic_runner.py): PPO runner with reward perturbation support.
- [`demoscript_cartpole.py`](src/experiment_3_gymnasium_noise/demoscript_cartpole.py): compact paired demo.

### `src/external/All-KAN`

The KAN wrappers use selected implementations from the cloned [`All-KAN`](src/external/All-KAN/README.md) repository. If this folder is missing in a fresh checkout, clone it with:

```bash
git clone https://github.com/hoangthangta/All-KAN.git src/external/All-KAN
```

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

The main dependencies are PyTorch, NumPy, Matplotlib, tqdm, and Gymnasium with classic-control/Box2D extras.

## Running the Experiments

The protocol files are the main entry points. They define the sweeps near the top of each file, then delegate to the corresponding runner.

```bash
python src/experiment_1_feynman/experiment_protocol_main.py
python src/experiment_2_gymnasium/experiment_classic_protocol_main.py
python src/experiment_3_gymnasium_noise/experiment_classic_protocol_main.py
```

Notes:

- These are full sweep scripts and can be expensive. Check `n_threads`, seed counts, sample sizes, and episode budgets before launching.
- The Feynman protocol currently has `ACTIVE_MODEL_GROUPS = ["mlp_matched"]` in the source. Include `"kan"` when running the full KAN plus matched-MLP sweep.
- Each runner writes timestamped CSVs and maintains a `latest_result_path.txt` pointer in its experiment output directory.
- The runners are designed to resume from existing CSVs and prune incomplete checkpoint rows.

## Recreating Figures

Committed paper figures live under `results/` and `_latex_overleaf/figures/`.

Experiment 1 and 2 include raw CSVs in `results/`:

```bash
python src/experiment_1_feynman/visualization_results.py
python results/experiment_1_feynman/visualizations/figure_1.py
python results/experiment_1_feynman/visualizations/figure_2.py
python results/experiment_2_gymnasium/visualizations/figure_1.py
python results/experiment_2_gymnasium/visualizations/figure_2.py
```

## Results Files

The main committed outputs are:

- `results/experiment_1_feynman/*.csv`: raw Feynman benchmark results.
- `results/experiment_1_feynman/Figure_X1_1.pdf` and `Figure_X1_2.pdf`: Feynman paper figures.
- `results/experiment_2_gymnasium/*.csv`: raw Gymnasium Classic Control PPO results.
- `results/experiment_2_gymnasium/Figure_X2_1.pdf` and `Figure_X2_2.pdf`: Gymnasium paper figures.
- `results/experiment_3_gymnasium_noise/Figure_X3_3.pdf`: noisy-reward paper figure.

The LaTeX project copies the final figures into `_latex_overleaf/figures/`.

## License

This project is released under the license in [`LICENSE`](LICENSE). The vendored All-KAN code has its own license in [`src/external/All-KAN/LICENSE`](src/external/All-KAN/LICENSE).

## Citation

If you find this paper useful, please cite us:
```
@inproceedings{riehlsample,
  title={Sample-Efficiency of Kolmogorov--Arnold Networks},
  author={Riehl, Kevin and El-Baklish, Shaimaa K and Wu, Fan and Kouvelas, Anastasios},
  booktitle={AXIOM: Foundations of Efficient Deep Learning, NeurIPS2026}
}
```