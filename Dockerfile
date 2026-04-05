# Use a standard Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables for non-interactive installs and output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PowerGridworld (OpenDSS and building tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# 1. Install primary requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy and install PowerGridworld physics engine
# This allows the container to have the core simulator locally
COPY powergrid-main/PowerGridworld-main /app/PowerGridWorld
RUN cd /app/PowerGridWorld && pip install -e .

# 3. Copy the OpenEnv task code and Dashboard assets
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY openenv.yaml /app/openenv.yaml
COPY baseline_inference.py /app/baseline_inference.py
COPY dashboard_server.py /app/dashboard_server.py
COPY train_mappo.py /app/train_mappo.py
COPY solar_grid_schematic.png /app/solar_grid_schematic.png
COPY README.md /app/README.md

# 4. Final configuration for OpenEnv evaluation
# Ensure the src directory is in the PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose the correct port for Hugging Face Spaces
EXPOSE 7860

# Default command: Start the real-time interactive dashboard
CMD ["python", "dashboard_server.py"]
