# orbital-decision-mesh

# Orbital Decision Mesh (`orbital-decision-mesh`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%252b-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Server-green.svg)](https://github.com/PrefectHQ/fastmcp)

`orbital-decision-mesh` is a production-grade, high-performance simulation and decision-making platform designed for **Low Earth Orbit (LEO) satellite constellation beam-hopping allocation**. It merges **Multi-Agent Reinforcement Learning (MAPPO)** for real-time policy control with an **agentic LLM "Slow Loop"** via the Model Context Protocol (FastMCP) and comprehensive enterprise observability.

---

## 🏗️ System Architecture

The platform separates real-time execution from agentic reasoning and telemetry tracking:

1. **The Fast Loop (Real-Time Control):**
   - **Environment (`SatComBeamEnv`):** Multi-agent Gymnasium environment simulating satellite orbital dynamics via Skyfield OMM parsers and dynamic cell demand matrices.
   - **MAPPO Policy (`MAPPOPolicy`):** Decentralized Actor / Centralized Critic architecture (`obs_dim: 12`, `act_dim: 8`, `global_state_dim: 48`) optimizing beam-to-cell allocations with Generalized Advantage Estimation (GAE).

2. **The Slow Loop (Agentic Ground Copilot):**
   - **FastMCP Server (`src/mcp/server.py`):** Exposes live satellite telemetry, orbital visibility calculations, and resource allocations to LLM ground assistants using the Model Context Protocol.
   - **RAG & Reasoning:** Ground station advisory copilot for operational intervention and telemetry analysis.

3. **Agentic Observability & MLOps:**
   - **Evidently AI:** Real-time data and concept drift detection on streaming orbital features.
   - **LangSmith:** Token tracing and prompt audit tracking for the slow-loop copilot.
   - **Prometheus:** System metric emitters for telemetry monitoring and anomaly detection triggering automated Continuous Training (CT) pipelines.

---

## 📂 Repository Structure

```text
orbital-decision-mesh/
├── .github/
│   └── workflows/          # CI/CD pipelines (linting, test execution)
├── airflow/
│   └── dags/               # Airflow DAGs for automated CT triggers
├── configs/
│   ├── env_config.yaml     # Simulation parameters and ground stations
│   └── mappo_config.yaml   # MAPPO hyperparameters (lr, clip, GAE lambda)
├── data/
│   └── starlink_cache.json # Cached OMM telemetry for offline execution
├── scripts/
│   ├── fetch_tle.py        # Space-Track ingestion and fallback utility
│   ├── train.py            # MAPPO end-to-end training execution loop
│   └── run_mcp.py          # FastMCP server runner for the slow loop
├── src/
│   ├── data/loader.py      # Space-Track API client & SGP4 OMM parser
│   ├── envs/sat_env.py     # Multi-agent Gymnasium environment
│   ├── models/mappo_policy.py # Actor-Critic networks & PPO vectorized updates
│   ├── observability/      # Prometheus metrics & Evidently drift hooks
│   └── mcp/server.py       # FastMCP server endpoints
├── tests/                  # Unit and integration test suite
├── .env.example            # Environment variable template
├── pyproject.toml          # Project build metadata & dependencies
└── LICENSE                 # Apache 2.0 License
```

---
##  Getting Started Locally
1. Clone & Set Up Virtual Environment

```Bash
# Clone the repository
git clone [https://github.com/yourusername/orbital-decision-mesh.git]

# Navigate to clone's root
cd orbital-decision-mesh

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate 

# Upgrade pip and setup tools
python -m pip install --upgrade pip
```

2. Install Dependencies



```Bash
# Install the package in editable mode with all core requirements via Hatchling:
pip install -e .
```

3. Configure Environment Variables

```Bash
# Copy the configuration template and populate your Space-Track credentials(optional)
cp .env.example .env
```

4. Run Sanity Checks & Training

```Bash
#Test Telemetry Ingestion:
python scripts/fetch_tle.py

#Execute MAPPO Training Loop:
python scripts/train.py
```



# 🛠️ Tech Stack & Libraries

Core Language: Python 3.10+

Build System: Hatchling (pyproject.toml)

Reinforcement Learning: PyTorch, Gymnasium

Orbital Mechanics: Skyfield, SGP4

Agentic Interface: FastMCP

Observability: Prometheus, Evidently, LangSmith

# 📄 License
This project is licensed under the terms of the Apache 2.0 License. See the LICENSE file for details.