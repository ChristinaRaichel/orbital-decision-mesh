# Orbital Decision Mesh

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Protocol: FastMCP](https://img.shields.io/badge/FastMCP-Server-purple.svg)](https://github.com/modelcontextprotocol/python-sdk)

An enterprise-grade, multi-agent reinforcement learning (MARL) and agentic control platform engineered for **Low Earth Orbit (LEO) satellite beam-hopping optimization**. 

This system bridges high-frequency orbital physics simulation with low-frequency agentic reasoning, utilizing a **Fast Loop** for real-time multi-agent resource allocation and a **Slow Loop** for RAG-backed LLM copilot oversight, model context protocol (MCP) server tooling, and continuous MLOps observability.

---

## System Architecture

The platform separates real-time multi-agent control from higher-level cognitive oversight and monitoring:

```text
       [ Space-Track OMM API / TLE Ingestion ]
                       │
                       ▼
         ┌───────────────────────────┐
         │  Fast Loop (Real-Time)    │
         │  - SatComBeamEnv (Gym)    │
         │  - MAPPO Actor-Critic     │
         │  - Beam Hop Allocation    │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[ Evidently / Prometheus ]   [ FastMCP Gateway ]
 (Drift & System Metrics)            │
                                     ▼
                             ┌───────────────────┐
                             │ Slow Loop (Agent) │
                             │ - RAG Telemetry   │
                             │ - Ground Copilot  │
                             │ - Airflow CT Trig │
                             └───────────────────┘

```

## Key Technical Highlights
### 1. Robust Data Pipeline & Orbital Mechanics
* Native OMM Parsing: 
Bypasses legacy TLE string column-offset corruptions and Alpha-5 catalog ID overflows by ingesting structured Orbital Mean Elements Messages (EarthSatellite.from_omm) via Skyfield.
* Deterministic Fallback: 
Includes automated fallback snapshots (starlink_cache.json) for air-gapped test environments and CI/CD pipelines.
### 2. Fast Loop: Multi-Agent Reinforcement Learning (MAPPO)
* Decentralized Execution, Centralized Training (CTDE): 
Implements a Multi-Agent PPO architecture where individual satellite agents observe local states ($obs\_dim = 12$) and select discrete beam targets ($act\_dim = 8$), while a centralized critic evaluates global constellation state ($global\_state\_dim = 48$).
* Optimized Advantage Estimation: 
Features vectorized Generalized Advantage Estimation (GAE), clipped policy surrogate objectives, and stable tensor gradient flows.
### 3. Slow Loop: Agentic Observability & FastMCP Gateway
* FastMCP Server Integration: 
Exposes live constellation telemetry, RAG historical logs, and MAPPO policy triggers as native Model Context Protocol tools.
* Ground Telemetry Copilot: 
An LLM-backed operational assistant that interprets metric threshold breaches, analyzes orbital geometry blockages, and suggests gateway handover or beam redistribution strategies.
### 4. Enterprise MLOps & Continuous Training (CT)
* Data & Concept Drift: 
Integrated Evidently AI monitoring pipelines to detect telemetry shifts and trigger automated Airflow Continuous Training (CT) pipelines.
* Tracing & Telemetry: 
LangSmith token tracking and Prometheus metric emitters for real-time production visibility.

## 📁 Repository Directory 

```StructurePlaintextc
orbital-decision-mesh/
├── .github/workflows/      # CI/CD test execution pipelines
├── airflow/dags/           # Airflow DAGs for automated CT triggers
├── configs/                # YAML configurations (env & MAPPO hyperparameters)
├── data/                   # Cached OMM telemetry snapshots
├── scripts/                # Execution entrypoints (fetch_tle.py, train.py, run_mcp.py)
├── src/
│   ├── agents/             # Slow loop Ground Copilot & prompt templates
│   ├── data/               # Space-Track API loader & OMM parser
│   ├── envs/               # Gymnasium multi-agent satellite environment
│   ├── mcp/                # FastMCP server integration
│   ├── models/             # MAPPO Actor-Critic policy architecture
│   └── observability/      # Evidently drift & Prometheus metrics hooks
├── tests/                  # Comprehensive unit & integration test suite
├── .env.example            # Environment credential templates
├── pyproject.toml          # Modern Hatchling build backend & dependencies
└── LICENSE                 # Apache 2.0 License
```


## Installation & Quickstart
### 1. Clone & Setup Environment
```Bash
git clone [https://github.com/yourusername/orbital-decision-mesh.git](https://github.com/yourusername/orbital-decision-mesh.git)
cd orbital-decision-mesh

# Copy environment variables template
cp .env.example .env
```

### 2. Install Dependencies
   Using modern Python packaging (pyproject.toml with Hatchling):
   ```Bash
   pip install -e .
   ```
### 3. Run Verification & Test Suite
``` Bash
# Run unit and integration tests across fast and slow loops
pytest tests/ -v
```

### 4. Execute WorkflowsBash
```
Fetch live TLE/OMM telemetry and cache locally
python scripts/fetch_tle.py

# Run MAPPO training loop (Fast Loop)
python scripts/train.py

# Launch FastMCP server gateway (Slow Loop)
python scripts/run_mcp.py

```
## License
This project is licensed under the terms of the Apache 2.0 License. See the LICENSE file for details.