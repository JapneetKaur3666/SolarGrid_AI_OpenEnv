import os
import json
import openai
from typing import Dict, Any
from src.envs.openenv_wrapper import SolarGridEnv
from src.envs.models import SolarObservation, SolarAction
import streamlit as st

st.title("⚡ SolarGrid AI")
st.write("Smart Solar Grid Optimization System")

st.success("App is running 🚀")

# PRE-SUBMISSION CONFIGURATION
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN") # No default as per requirements

def get_openai_client():
    # Grader expects configuring via API_BASE_URL and HF_TOKEN (or generic API key)
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL
    )

client = get_openai_client()

def get_agent_action(obs: SolarObservation, task_name: str) -> SolarAction:
    """Uses LLM to decide on the best dispatcher action."""
    system_prompt = f"""
    You are an AI Grid Dispatcher for a Smart Solar Grid. 
    Task: {task_name}
    Current Observation: {obs.model_dump_json()}
    
    You must output a JSON object conforming to the following SolarAction format:
    {{
        "charge_discharge_rate": float (-1.0 to 1.0),
        "shed_non_critical_load": bool,
        "grid_export_limit": float
    }}
    - Charge battery (>0) if solar surplus is high and price is low.
    - Discharge (<0) if demand is peak and price is high.
    - Shed load if voltage is too high or grid is stressed.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return SolarAction(**data)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Default fallback action
        return SolarAction(charge_discharge_rate=0.0, shed_non_critical_load=False, grid_export_limit=1.0)

def evaluate_task(task_id: str, num_steps: int = 10):
    """Evaluates the agent on a specific task scenario."""
    env = SolarGridEnv(task_id=task_id)
    obs = env.reset()
    
    total_reward = 0
    print(f"[START] task={task_id}", flush=True)
    
    for i in range(num_steps):
        action = get_agent_action(obs, task_id)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.total_reward
        current_step = i + 1
        
        # Display the 0.0-1.0 task score from the grader
        score = info.get("task_score", 0.0)
        print(f"[STEP] step={current_step} reward={reward.total_reward:.4f}", flush=True)
        
        if terminated or truncated:
            break
            
    final_score = info.get("task_score", 0.0)
    print(f"[END] task={task_id} score={final_score:.4f} steps={env.current_step}", flush=True)
    return final_score

if __name__ == "__main__":
    tasks = ["maximize-self-consumption", "peak-shaving", "emergency-load-shedding"]
    results = {}
    
    for task in tasks:
        results[task] = evaluate_task(task)
        
    print("\n--- Final Hackathon Baseline Summary ---", flush=True)
    for task, score in results.items():
        print(f"* Task: {task} | Score: {score:.4f}", flush=True)
