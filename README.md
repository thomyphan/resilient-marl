# Resilient Multi-Agent Reinforcement Learning

- Featured algorithms:
    - Randomized Adversarial Training (RAT) [1]
	- Resilient Adversarial value Decomposition with Antagonist-Ratios (RADAR) [1]

## 1. Implemented domains

All available domains are listed in the table below. The labels are used for the commands below (in 4. and 5.).

| Domain   | Label            | Description                                                       |
|----------|------------------|-------------------------------------------------------------------|
| PP[5,2]  | `PredatorPrey-2` | Predator prey domain with 2 predators and 1 prey in a 5x5 grid.   |
| PP[7,4]  | `PredatorPrey-4` | Predator prey domain with 4 predators and 2 preys in a 7x7 grid.  |
| PP[10,8] | `PredatorPrey-8` | Predator prey domain with 8 predators and 4 preys in a 10x10 grid.|

## 2. Implemented algorithms

The available base algorithms and their variants are listed in the table below. In addition, we implemented `COMA`&ast;, `MADDPG`, and `M3DDPG`&ast; as general baselines. The labels are used for the command below (in 5.).

|                                    | Value-based | Standard Actor-Critic | Proximal Policy Optimization |
|------------------------------------|-------------|-----------------------|------------------------------|
| **Independent Learning**           | `DQN`*      | `IAC`*                | `PPO`*                       |
| **Independent RAT variant**        | `RAT_DQN`   | `RAT_IAC`             | `RAT_PPO`                    |
| **Linear Value Decomposition**     | `VDN`*      | `RADAR` or `RADAR_X`**| `RADAR_PPO`                  |
| **Non-Linear Value Decomposition** | `QMIX`*     | `AC-QMIX`             | `PPO-QMIX`                   |

*no explicit training of adversaries

**RADAR variant with fixed adversary ratio

## 3. Experiment parameters

The experiment parameters like the number of time steps used for training (`nr_steps`) or the number of episodes per phase (`params["test_interval"]`) are specified in `settings.py`. All other hyperparameters are set in the corresponding python modules in the package `radar/agents`, where all final values as listed in the technical appendix are specified as default value.

All hyperparameters can be adjusted by setting their values via the `params` dictionary in `settings.py`.

## 4. Generating test cases

To generate a test case with `RADAR_X`, domain `D` (see table in 1.), and adversary ratio `A`, run the following command:

    python generate_test.py RADAR_X D A

This command will create a folder with the name pattern `tests/N-agents_domain-D_adversaryratio-D_RADAR_X_datetime` which contains the trained models saved in `protagonist_model.pth` and `adversary_model.pth`.

## 5. Running MARL algorithm with online evaluation

To perform a training run with MARL algorithm `M` (see table in 2.), domain `D` (see table in 1.), and adversary ratio `A`, run the following command:

    python training_run.py M D A

This command will create a folder with the name pattern `output/N-agents_domain-D_adversaryratio-D_M_datetime` which contains the trained models saved in `protagonist_model.pth` and `adversary_model.pth`, the training progress data saved in `returns.json`, and the environment state data saved in `episode_x.json`.

The file `returns.json` also contains the full test results which were obtained during training which can be plotted using any visualization library (e.g., `matplotlib` or `seaborn`). All test cases are automatically loaded from the folder `tests`, where all previously generated test cases are stored per default (see 4.).

## 6. Setting the random seed

To manually set the random seed, uncomment the corresponding block in `settings.py`.

## References

- [1] T. Phan et al., ["Resilient Multi-Agent Reinforcement Learning with Adversarial Value Decomposition"](https://ojs.aaai.org/index.php/AAAI/article/view/17348/17155), in AAAI 2021
