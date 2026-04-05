# Implementation Plan: PowerGrid-OpenEnv Solar Hackathon

This plan outlines the steps required to build and deploy the **Solar Grid Routing** project for the India Mega AI Hackathon.

## Phase 1: Environment Setup & Core Simulation (Days 1–3)
- [ ] **Infrastructure Setup**: Clone `PowerGridworld` and verify installation with existing tests.
- [ ] **Solar Data Integration**: Prepare Time-Series data for PV (Photovoltaic) generation and demand profiles.
- [ ] **Grid Configuration**: Define a multi-node solar grid layout (residential & commercial nodes) using the `PowerGridworld` standard YAML/JSON format.

## Phase 2: PettingZoo MARL Wrapper (Days 4–7)
- [ ] **Agent Mapping**: Define which components are "agents" (batteries, loads, solar nodes).
- [ ] **Observation/Action Definition**:
  - *Observation*: State of charge (SOC), local net power, grid voltage/frequency.
  - *Action*: Charge/Discharge rate, load shedding level, buy/sell power from the grid.
- [ ] **PettingZoo Parallel API Implementation**:
  - Inherit from `pettingzoo.ParallelEnv`.
  - Map `reset()` and `step()` to `PowerGridworld`'s simulation engine.
  - **Reward Engineering**: Design a composite reward for cost, peak shaving, and stability.

## Phase 3: RL Training & Optimization (Days 8–11)
- [ ] **Baseline Training**: Train a single-agent PPO (Proximal Policy Optimization) using `stable-baselines3` (or `tianshou`).
- [ ] **Multi-Agent Training**: Implement **MAPPO** (Multi-Agent PPO) to optimize energy routing across all nodes.
- [ ] **Hyperparameter Tuning**: Optimize learning rates, entropy coefficients, and reward weighting.
- [ ] **Evaluation**: Benchmark the trained policies against simple heuristics (e.g., "always charge when solar is peaking").

## Phase 4: OpenEnv Standardization & Packaging (Finishing)
- [ ] **OpenEnv Integration**: Wrap the PettingZoo environment into a standard `Gymnasium` interface compatible with OpenEnv's requirements.
- [ ] **Dockerfile Creation**: Build a Docker image containing:
  - Mini-Conda/Python environment.
  - `PowerGridworld` and PettingZoo dependencies.
  - Trained RL models (weights).
- [ ] **OpenEnv Scoring**: Test the environment using `openenv-eval` (or equivalent tool) to ensure it correctly scores on Meta's benchmarks.
- [ ] **Hugging Face Deployment**: Upload the environment and model to the Hugging Face Hub under the OpenEnv tags.

## 🏆 Final Deliverables
1. **GitHub Repository**: Complete source code, structure, and documentation.
2. **OpenEnv Docker Image**: Container ready for evaluation by Meta/Hugging Face.
3. **Project Presentation**: Demonstrating how MARL improves solar grid efficiency and routing.

---
> [!TIP]
> Focus on **Peak Shaving** as a primary reward metric — it's highly relevant for grid stability and easy to demonstrate.
