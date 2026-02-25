# Wind Farm Control

Multi-agent reinforcement learning for wind farm yaw control optimization using FLORIS and MAPPO.

## Project Structure

```
wind-farm-control/
├── src/
│   └── wind_farm_control/        # Main package
│       ├── environments/         # Environment implementations
│       │   └── floris_env.py    # Multi-agent FLORIS environment
│       ├── models/               # Neural network models
│       │   ├── actor.py         # Actor network (policy)
│       │   └── critic.py        # Critic network (value function)
│       ├── training/             # Training utilities
│       │   └── mappo_trainer.py # MAPPO training functions
│       ├── evaluation/           # Evaluation utilities
│       │   └── evaluator.py     # Performance evaluation & plotting
│       └── utils/                # Utility functions
│           └── hyperparameter_tuning.py  # Optuna integration
│
├── tests/                        # Unit tests
│   ├── test_environment.py      # Environment tests
│   ├── test_models.py           # Model tests
│   └── test_training.py         # Training tests
│
├── notebooks/                    # Jupyter notebooks
│   ├── test_mappo.ipynb         # MAPPO training & evaluation
│   └── test_controllers.ipynb   # Controller testing
│
├── data_generation/              # Data generation scripts
│   └── farm_types/              # FLORIS configuration files
│
├── testing_controllers/          # Legacy single-agent controllers
│   ├── floris_env.py            # Gymnasium environment
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
│
├── runs/                         # Training run outputs
├── archive/                      # Archived code
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Installation

```bash
# Install the package in editable mode with dev dependencies
uv pip install -e ".[dev,tuning]"

# Or using pip
pip install -e ".[dev,tuning]"
```

## Quick Start

### Training a MAPPO Agent

```python
from wind_farm_control.training import train_mappo

# Train the agent
agent, env, trainer = train_mappo(
    config_path="data_generation/farm_types/gch.yaml",
    timesteps=50000,
    learning_rate=5e-4
)
```

### Evaluating Performance

```python
from wind_farm_control.evaluation import evaluate_mappo_vs_baseline, plot_mappo_performance

# Evaluate against baseline
results = evaluate_mappo_vs_baseline(agent, env, n_episodes=20)
print(results["Gain_%"].describe())

# Visualize results
plot_mappo_performance(results)
```

### Hyperparameter Tuning

```python
from wind_farm_control.utils import run_hyperparameter_tuning

# Run Optuna optimization
study = run_hyperparameter_tuning(
    config_path="data_generation/farm_types/gch.yaml",
    n_trials=10,
    training_timesteps=5000
)
print(f"Best parameters: {study.best_params}")
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py

# Run with coverage
pytest --cov=src tests/
```

## Key Features

- **Multi-Agent Environment**: PettingZoo-compatible FLORIS environment for cooperative yaw control
- **MAPPO Algorithm**: State-of-the-art multi-agent PPO with centralized critic
- **Parameter Sharing**: All agents share the same policy and value networks
- **Hyperparameter Tuning**: Integrated Optuna support for automatic optimization
- **Comprehensive Testing**: Unit tests for all major components

## Environment Details

### Observation Space
- Wind direction: [260°, 290°]
- Wind speed: [5 m/s, 15 m/s]
- Turbulence intensity: [0.03, 0.25]

### Action Space
- Yaw angle adjustment: [-1, 1] (scaled to [-25°, 25°])

### Reward
- Global reward: Total farm power output (encourages cooperation)

## Contributing

1. Add new features in the appropriate `src/` subdirectory
2. Write tests in `tests/` for new functionality
3. Update documentation as needed
4. Run tests before committing: `pytest tests/`

## License

[Add your license here]
