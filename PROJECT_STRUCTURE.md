# PowerGrid-OpenEnv: Solar Grid Routing Hackathon Proposal

## Overview
This project aims to simulate a solar power grid using **PowerGridworld**, structure it as a Multi-Agent Reinforcement Learning (MARL) environment using **PettingZoo**, and standardize the evaluation and deployment using **OpenEnv** as part of the Meta/Hugging Face India Mega AI Hackathon.

## Core Idea
- **PowerGridworld**: A lightweight, modular simulation for power flow and grid-level variables.
- **PettingZoo**: A unified API for multi-agent RL (MARL). We wrap PowerGridworld to treat each solar node, battery, and household as an active agent.
- **Optimal Routing Strategies**: Reinforcement Learning (e.g., MAPPO, PPO) to learn energy distribution that minimizes costs and peaks.
- **OpenEnv**: Final standardization to allow Meta/Hugging Face to evaluate the model in a containerized environment (Docker).

## 📁 Proposed Project Structure

```text
├── .github/                # CI/CD Workflows
├── PowerGridworld/         # Core simulator (submodule or vendor)
├── src/
│   ├── envs/               # Environment wrappers
│   │   ├── __init__.py
│   │   ├── powergrid_wrapper.py   # Main simulation logic integration
│   │   ├── pettingzoo_impl.py      # PettingZoo API implementation
│   │   └── openenv_wrapper.py     # OpenEnv/Gymnasium standardization
│   ├── agents/             # RL Algorithms
│   │   ├── __init__.py
│   │   ├── ppo_agent.py           # Single-agent baseline
│   │   └── mappo_agent.py         # Multi-agent PPO for routing
│   ├── training/           # Training scripts
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/              # Helper functions
├── config/                 # Environment and Agent configurations
│   ├── grid_config.yaml
│   └── rl_config.yaml
├── notebooks/              # Exploratory notebooks
├── Dockerfile              # OpenEnv containerization
├── requirements.txt
├── README.md               # Project overview
└── ARCHITECTURE.md         # Detailed system design
```

## 🛠️ Tech Stack
- **Languages**: Python 3.10+
- **Simulator**: PowerGridworld (NREL)
- **MARL Interface**: PettingZoo (Parallel API)
- **RL Frameworks**: CleanRL or Stable-Baselines3 (customized for MARL)
- **Standardization**: OpenEnv (Hugging Face / Meta)
- **Containerization**: Docker
