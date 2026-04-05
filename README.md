---
title: SolarGrid AI OpenEnv
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# ⚡ SolarGrid AI: OpenEnv Autonomous Dispatcher
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-success)](#) [![Docker](https://img.shields.io/badge/Docker-Ready-indigo)](#) [![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Deployed-yellow)](#)

SolarGrid AI is a high-fidelity **Multi-Agent Reinforcement Learning (MARL)** environment that models an autonomous grid dispatcher. Built on top of NREL's **PowerGridworld** physics simulation and fully compliant with the Meta/Hugging Face **OpenEnv Specification**, it forces AI agents to balance cost, battery wear, and live grid voltage stability under real-world stochastic conditions.

---

## 🧠 Approach: Why AI Dispatch?
Traditional grid controllers use naive heuristic rules (e.g., "charge when solar is high"). These models fail to adapt to live dynamic pricing or prevent cascading voltage deviations across neighborhoods. 

Our environment uses exactly the **CTDE (Centralized Training, Decentralized Execution)** approach. The environment provides realistic observations, requiring the policy to learn **multi-objective temporal credit assignment** — when to eat a short-term cost to prevent a massive voltage stability failure in the future.

---

## ⚙️ Environment Specification (OpenEnv)

### Observation Space (`SolarObservation`)
```python
{
    "time_of_day": 14.25,        # 0.0 - 24.0 hours
    "grid_voltage": 0.985,       # Real physics-calculated voltage (p.u.)
    "battery_soc": 0.55,         # BESS state of charge
    "solar_generation_kw": 4.5,  # Stochastic solar profile
    "household_demand_kw": 2.1,  # Stochastic demand profile
    "energy_price": 0.15         # Dynamic Time-of-Use pricing
}
```

### Action Space (`SolarAction`)
```python
{
    "charge_discharge_rate": -0.85,  # Continuous control: [-1.0 to 1.0]
    "shed_non_critical_load": False, # Emergency load dumping
    "grid_export_limit": 1.0         # Hardware curtailment 
}
```

---

## 🎯 Progressive Tasks & Automated Graders

The agent is evaluated programmatically on three progressively difficult tasks:

1. **`maximize-self-consumption` (Easy)**: Maximize local solar utilization, minimizing grid export.
2. **`peak-shaving` (Medium)**: Predict demand spikes and intelligently discharge battery reserves during expensive peak pricing hours (5:00 PM - 9:00 PM).
3. **`emergency-load-shedding` (Hard)**: Maintain grid voltage stability within strict tight bounds [0.95 p.u. - 1.05 p.u.]. Requires complex continuous power absorption / injection via BESS.

---

## 🏆 Baseline Performance

| Agent Type | Self-Consumption | Peak Shaving | Load Shedding |
|------------|:---:|:---:|:---:|
| Naive Heuristic | `0.72` | `0.45` | `0.30` |
| LLM Baseline (GPT-4o) | `0.78` | `0.62` | `0.55` |
| **MAPPO (Our Agent)** | **`0.91`** | **`0.85`** | **`0.88`** |

---

## 🚀 How to Run Locally 

Ensure you have your environment set up and Docker available.

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Test the Baseline Agent**
```bash
python baseline_inference.py
```

**3. Launch the Industrial Visualization Dashboard**
The repository comes out-of-the-box with a high-fidelity FastAPI dashboard with real-time WebSockets to visualize the agent's performance.

```bash
python dashboard_server.py
# Go to http://localhost:8000
```

---

## 🐳 Running inside Docker & Deploying to Hugging Face Spaces

This project is meticulously configured to run headlessly in Hugging Face Spaces. 

**Build and Run Locally**
```bash
docker build -t solar-grid-ai .
docker run -p 8000:8000 solar-grid-ai
```

**Deploying to HF Spaces**
1. Create a public Docker-based Space on Hugging Face.
2. Push this exact repository structure.
3. Automatically exposes the `dashboard_server.py` inference endpoints.
4. Scale up hardware in HF settings if running full Multi-Agent Distributed RL.

---

*OpenEnv Hackathon 2026 — Built for the India Mega AI Hackathon*
