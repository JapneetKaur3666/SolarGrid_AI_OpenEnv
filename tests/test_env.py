import pytest
import numpy as np
from src.envs.openenv_wrapper import SolarGridEnv
from src.envs.models import SolarObservation, SolarAction, SolarReward

def test_pydantic_model_compliance():
    """Ensures our Pydantic models are functioning correctly."""
    obs_data = {
        "time_of_day": 12.0,
        "grid_voltage": 1.0,
        "battery_soc": 0.5,
        "solar_generation_kw": 5.0,
        "household_demand_kw": 3.0,
        "energy_price": 0.15
    }
    obs = SolarObservation(**obs_data)
    assert obs.time_of_day == 12.0
    assert isinstance(obs, SolarObservation)

def test_env_reset_protocol():
    """Verifies that reset() returns a SolarObservation."""
    env = SolarGridEnv(task_id="maximize-self-consumption")
    obs = env.reset()
    assert isinstance(obs, SolarObservation)
    assert 0.0 <= obs.battery_soc <= 1.0

def test_env_step_protocol():
    """Verifies that step() returns (Obs, Reward, Terminated, Truncated, Info)."""
    env = SolarGridEnv(task_id="maximize-self-consumption")
    env.reset()
    
    action = SolarAction(charge_discharge_rate=0.5, shed_non_critical_load=False, grid_export_limit=1.0)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, SolarObservation)
    assert isinstance(reward, SolarReward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "task_score" in info
    assert "task_id" in info

@pytest.mark.parametrize("task_id", ["maximize-self-consumption", "peak-shaving", "emergency-load-shedding"])
def test_task_scores_in_range(task_id):
    """Ensures task_score is always between 0.0 and 1.0 for all missions."""
    env = SolarGridEnv(task_id=task_id)
    env.reset()
    
    for _ in range(5):
        action = SolarAction(charge_discharge_rate=0.1, shed_non_critical_load=False, grid_export_limit=1.0)
        _, _, _, _, info = env.step(action)
        score = info["task_score"]
        assert 0.0 <= score <= 1.0, f"Task score {score} for {task_id} is out of 0.0-1.0 range"

def test_reward_signal_generation():
    """Checks if the internal reward components are being calculated."""
    env = SolarGridEnv(task_id="maximize-self-consumption")
    env.reset()
    
    action = SolarAction(charge_discharge_rate=-1.0, shed_non_critical_load=True, grid_export_limit=0.0)
    _, reward, _, _, _ = env.step(action)
    
    # Total reward should be the sum of its parts
    assert reward.total_reward == (reward.cost_saving + reward.solar_utilization + reward.grid_stability)
