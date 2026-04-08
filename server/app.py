import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.envs.openenv_wrapper import SolarGridEnv
from src.envs.models import SolarAction

app = FastAPI()

# Solar Grid Environment Instance
env = SolarGridEnv(task_id="peak-shaving")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def get():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/circuit")
async def get_circuit():
    with open(os.path.join(BASE_DIR, "templates", "circuit.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/solar_grid_schematic")
async def get_schematic():
    img_path = os.path.join(BASE_DIR, "solar_grid_schematic.png")
    return FileResponse(img_path, media_type="image/png")

# --- OPENENV HACKATHON GRADER API ENDPOINTS ---
@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step_env(action: dict):
    # Parse incoming JSON dict to Pydantic SolarAction
    act = SolarAction(**action)
    obs, reward, terminated, truncated, info = env.step(act)
    return {
        "observation": obs.model_dump(),
        "reward": reward.total_reward,
        "done": terminated or truncated,
        "info": info
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    obs = env.reset()
    
    try:
        while True:
            # 1. Simulate an Agent decision (Simplified for front-end demo)
            # In a real run, this would call your baseline_inference.py logic
            if obs.solar_generation_kw > obs.household_demand_kw:
                action = SolarAction(charge_discharge_rate=0.8, shed_non_critical_load=False, grid_export_limit=1.0)
            else:
                action = SolarAction(charge_discharge_rate=-0.5, shed_non_critical_load=False, grid_export_limit=1.0)
            
            # 2. Step the Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Prepare the telemetry packet for the UI
            telemetry = {
                "step": env.current_step,
                "time": f"{int(obs.time_of_day):02d}:{int((obs.time_of_day % 1) * 60):02d}",
                "solar": round(obs.solar_generation_kw, 2),
                "demand": round(obs.household_demand_kw, 2),
                "battery_soc": round(obs.battery_soc * 100, 1),
                "voltage": round(obs.grid_voltage, 3),
                "price": round(obs.energy_price, 2),
                "score": round(info.get("task_score", 0) * 100, 1),
                "rewards": {
                    "cost": round(reward.cost_saving, 3),
                    "solar": round(reward.solar_utilization, 3),
                    "wear": round(reward.battery_wear, 3),
                    "stability": round(reward.stability_penalty, 3)
                },
                "action": "Charging" if action.charge_discharge_rate > 0 else "Discharging"
            }
            
            # 4. Push to Client
            await websocket.send_text(json.dumps(telemetry))
            
            # 5. Handle Reset
            if terminated or truncated:
                obs = env.reset()
                
            await asyncio.sleep(0.5) # Sim speed controller
            
    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    # Make sure we are in the right directory to find templates
    uvicorn.run(app, host="0.0.0.0", port=7860)
