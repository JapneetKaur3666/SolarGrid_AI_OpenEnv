from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class SolarObservation(BaseModel):
    """The full sensor state of the grid node/agent."""
    time_of_day: float = Field(..., description="Current time-of-day in hours (0–24)")
    grid_voltage: float = Field(..., description="Current voltage on the local bus (standard normalized 0.9–1.1)")
    battery_soc: float = Field(..., description="Current state of charge of the local battery (0.0–1.0)")
    solar_generation_kw: float = Field(..., description="Current local PV yield in kW")
    household_demand_kw: float = Field(..., description="Current local energy demand in kW")
    energy_price: float = Field(..., description="Current real-time energy price from the grid")

class SolarAction(BaseModel):
    """Actions the agent can take to manage energy routing."""
    charge_discharge_rate: float = Field(..., description="Rate of battery charge (>0) or discharge (<0) in kW (normalized -1.0 to 1.0)")
    shed_non_critical_load: bool = Field(..., description="Whether to temporarily shut off non-critical appliances to save grid capacity")
    grid_export_limit: float = Field(..., description="Maximum allowed power export to the grid in kW")

class SolarReward(BaseModel):
    """A detailed breakdown of the reward function for transparency and signal."""
    cost_saving: float = Field(..., description="Reward for minimizing electricity cost")
    solar_utilization: float = Field(..., description="Reward for using local solar vs grid")
    battery_wear: float = Field(..., description="Penalty for excessive charge/discharge cycling")
    stability_penalty: float = Field(..., description="Heavy penalty for voltage spikes or out-of-bounds")
    total_reward: float = Field(..., description="The composite scalar reward passed to the agent")
