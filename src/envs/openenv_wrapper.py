import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from src.envs.models import SolarObservation, SolarAction, SolarReward
from gridworld.multiagent_env import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.energy_storage.energy_storage_env import EnergyStorageEnv

class SolarGridEnv(gym.Env):
    """
    OpenEnv compliant wrapper for PowerGridworld.
    Simulates a Smart Grid Dispatcher real-world task.
    """
    def __init__(self, task_id: str = "maximize-self-consumption", config: Optional[Dict] = None):
        super().__init__()
        self.task_id = task_id
        
        # Default config if none provided
        if config is None:
            config = self._get_default_config()
            
        # Initialize the underlying PowerGridworld MultiAgentEnv
        self.underlying_env = MultiAgentEnv(
            common_config=config["common"],
            pf_config=config["pf"],
            agents=config["agents"]
        )
        self.agent_name = config["agents"][0]["name"]
        
        # OpenEnv metadata
        self.metadata = {"render_modes": ["ansi"]}
        self.current_step = 0
        self.max_steps = config["common"].get("max_episode_steps", 96)
        
        # Grader Tracking
        self.total_solar_gen = 0.0
        self.total_grid_export = 0.0
        self.peak_grid_draw = 0.0
        self.time_in_safe_voltage = 0
        self.last_action_rate = 0.0 # Tracking for Churn (Battery Wear)
        self.cumulative_task_score = 0.0

    def _get_default_config(self):
        # Basic configuration for 1 day of simulation using IEEE 13 node feeder
        return {
            "common": {
                "start_time": "2026-04-01 00:00:00",
                "end_time": "2026-04-02 00:00:00",
                "control_timedelta": pd.Timedelta(minutes=15),
                "max_episode_steps": 96
            },
            "pf": {
                "cls": OpenDSSSolver,
                "config": {
                    "feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
                    "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv"
                }
            },
            "agents": [
                {
                    "name": "dispatcher_0",
                    "bus": "671", 
                    "cls": EnergyStorageEnv,
                    "config": {
                        "name": "dispatcher_0",
                        "storage_range": (0.0, 100.0),
                        "initial_storage_mean": 50.0,
                        "max_power": 20.0
                    }
                }
            ]
        }

    def reset(self, seed=None, options=None) -> SolarObservation:
        super().reset(seed=seed)
        raw_obs = self.underlying_env.reset()
        self.current_step = 0
        
        # Reset trackers
        self.total_solar_gen = 0.0
        self.total_grid_export = 0.0
        self.peak_grid_draw = 0.0
        self.time_in_safe_voltage = 0
        self.last_action_rate = 0.0
        self.cumulative_task_score = 0.0
        
        return self._map_to_solar_observation(raw_obs[self.agent_name])

    def step(self, action: SolarAction) -> Tuple[SolarObservation, SolarReward, bool, bool, dict]:
        # 1. Map OpenEnv action to PowerGridworld action
        # EnergyStorageEnv expects a numpy array [-1, 1] for charge/discharge
        pg_action = {
            self.agent_name: np.array([action.charge_discharge_rate])
        }
        
        # 2. Step the simulation
        raw_obs, raw_rews, raw_dones, raw_info = self.underlying_env.step(pg_action)
        
        self.current_step += 1
        
        # 3. Update Grader Metrics
        obs = self._map_to_solar_observation(raw_obs[self.agent_name])
        self._update_grader_metrics(obs)
        
        # 4. Calculate Reward using the meaningful reward logic
        reward_breakdown = self._calculate_reward(raw_obs[self.agent_name], action)
        
        # 5. Calculate Task Score (0.0 - 1.0)
        task_score = self._calculate_task_score()
        
        # 6. Check termination
        terminated = raw_dones["__all__"]
        truncated = self.current_step >= self.max_steps
        
        # 7. Build Info including the task score
        info = {**raw_info[self.agent_name], "task_score": task_score, "task_id": self.task_id}
        
        return obs, reward_breakdown, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """OpenEnv specific: Returns global state."""
        return {
            "all_voltages": self.underlying_env.voltages,
            "step": self.current_step,
            "time": str(self.underlying_env.time)
        }

    def _map_to_solar_observation(self, agent_obs: np.ndarray) -> SolarObservation:
        # Map EnergyStorageEnv observations + global grid variables
        node_voltage = self.underlying_env.voltages.get("671.1", 1.0)
        
        # Mocking solar and demand based on time-of-day for the prototype
        hour = self.underlying_env.time.hour
        solar = max(0, 10 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        demand = 3.0 + 2.0 * np.sin(np.pi * (hour - 12) / 24)
        
        return SolarObservation(
            time_of_day=float(self.underlying_env.time.hour + self.underlying_env.time.minute/60.0),
            grid_voltage=float(node_voltage),
            battery_soc=float(agent_obs[0] / 100.0), # Assuming max storage is 100
            solar_generation_kw=float(solar),
            household_demand_kw=float(demand),
            energy_price=self._get_current_price()
        )

    def _get_current_price(self) -> float:
        # Real-world Time-of-Use price simulation
        hour = self.underlying_env.time.hour
        if 17 <= hour <= 21: # Peak hours 5 PM - 9 PM
            return 0.45 # High price
        return 0.15 # Low price

    def _calculate_reward(self, agent_obs: Any, action: SolarAction) -> SolarReward:
        # Example logic for the "Meaningful Reward Function"
        soc = agent_obs[0] / 100.0
        volt = self.underlying_env.voltages.get("671.1", 1.0)
        
        # Mocking solar and demand for reward
        hour = self.underlying_env.time.hour
        solar = max(0, 10 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        demand = 3.0 + 2.0 * np.sin(np.pi * (hour - 12) / 24)
        
        # 1. Cost Saving: Negative reward for grid import during high price
        price = self._get_current_price()
        net_load = demand - solar 
        cost_penalty = -max(0, net_load) * price
        
        # 2. Solar Utilization: Reward for using solar locally
        solar_util = min(solar, demand) * 0.1 
        
        # 3. FAILURE PENALTY: Battery Wear (Churn)
        churn = abs(action.charge_discharge_rate - self.last_action_rate)
        wear_penalty = -(churn ** 2) * 2.0 
        self.last_action_rate = action.charge_discharge_rate
        
        # 4. FAILURE PENALTY: Grid Stability Spikes
        out_of_bounds = max(0, 0.95 - volt) + max(0, volt - 1.05)
        stability_penalty = -(out_of_bounds ** 2) * 50.0 
            
        total = cost_penalty + solar_util + wear_penalty + stability_penalty
        
        return SolarReward(
            cost_saving=float(cost_penalty),
            solar_utilization=float(solar_util),
            battery_wear=float(wear_penalty),
            stability_penalty=float(stability_penalty),
            total_reward=float(total)
        )

    def _update_grader_metrics(self, obs: SolarObservation):
        # Tracking for scoring
        self.total_solar_gen += obs.solar_generation_kw
        net_grid = obs.household_demand_kw - obs.solar_generation_kw
        if net_grid < 0: # Exporting
            self.total_grid_export += abs(net_grid)
        
        # Track peak draw during peak hours for Task 2
        if 17 <= obs.time_of_day <= 21:
            self.peak_grid_draw = max(self.peak_grid_draw, max(0, net_grid))
            
        # Track voltage stability for Task 3
        if 0.95 <= obs.grid_voltage <= 1.05:
            self.time_in_safe_voltage += 1

    def _calculate_task_score(self) -> float:
        """Calculates 0.0 - 1.0 score based on task specific criteria with shaping."""
        step_score = 0.0
        
        if self.task_id == "maximize-self-consumption":
            if self.total_solar_gen == 0: step_score = 1.0
            else:
                step_score = (self.total_solar_gen - self.total_grid_export) / self.total_solar_gen
            
        elif self.task_id == "peak-shaving":
            # Target 2.0 kW max draw during peak
            # SHAPING: Gradually reduce the score based on how close we are to violating the peak
            penalty_shaping = max(0, (self.peak_grid_draw - 2.0) / 2.0)
            step_score = 1.0 - penalty_shaping
            
        elif self.task_id == "emergency-load-shedding":
            # Ratio of time voltage was stable
            step_score = self.time_in_safe_voltage / self.current_step
            
        # Update running cumulative score (Shaping Credit)
        self.cumulative_task_score = (self.cumulative_task_score * (self.current_step - 1) + step_score) / self.current_step
        return float(np.clip(self.cumulative_task_score, 0.0, 1.0))
