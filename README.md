# SatQuery AI

SatQuery AI is an agentic remote-sensing vision-language assistant for analyzing satellite imagery via natural-language text queries. The system automatically routes queries to specialist tools (e.g., Visual Question Answering, Change Detection, Optical+SAR Fusion, Region Grounding), processes imagery, and returns evidence-grounded answers with an execution trace.

## Project Structure

```text
SatQuery-AI/
│
├── backend/
│   ├── agent/             # Agentic planner, router and registry (LangGraph base)
│   ├── models/            # Specialist remote sensing model wraps (e.g., VQA)
│   ├── preprocessing/     # Satellite image registration, normalization, cropping
│   ├── validation/        # Formats, bands, dimensions and coregistration validators
│   ├── evidence/          # Overlays, overlays rendering, confidence, and reports
│   └── api/               # FastAPI endpoints, routes, and main application setup
│
├── datasets/              # Dataset registries and local samples
├── experiments/           # Fine-tuning and hyperparameter experiments
├── notebooks/             # Research & exploration notebooks (Colab-ready)
├── tests/                 # Backend and model pipeline tests
├── requirements.txt       # Dependencies
└── .env                   # Configuration variables
```

## Getting Started

### Installation

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API Server

Start the FastAPI server locally:
```bash
python3 -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

The interactive API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
