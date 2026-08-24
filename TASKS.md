# Master Task List — SatQuery AI

## Completed Milestones

- [x] **Phase 0: Project Structure Setup**
  - [x] Create folder layout (`backend/agent`, `backend/models`, `backend/validation`, etc.).
  - [x] Define `requirements.txt` and base project configuration (`config.py`).
  - [x] Set up virtual environment and install backend packages.
- [x] **Phase 1: Foundation & Single-Image VQA**
  - [x] Define specialist model base class (`base.py`).
  - [x] Create input validator (`validator.py`).
  - [x] Implement `RemoteSensingVQAModel` with spectral/pixel-analysis fallback.
  - [x] Build FastAPI routes (`vqa.py`, `main.py`).
  - [x] Test green (vegetation) and blue (water) mock images.
- [x] **Phase 2: Single-Image Scene Captioning & Text-Guided Grounding**
  - [x] Implement `RemoteSensingCaptionModel` (BLIP + offline fallback).
  - [x] Implement `RemoteSensingGroundingModel` (bounding-box output + fallback).
  - [x] Expose `/api/v1/caption` and `/api/v1/grounding` routes.
  - [x] Add verification tests for scene descriptions and coordinates.
- [x] **Phase 3: Bi-temporal Change Detection & Change VQA**
  - [x] Build registration compatibility checks (`backend/preprocessing/registration.py`).
  - [x] Implement `ChangeDetectionModel` (pixel-difference map and thresholding).
  - [x] Implement `ChangeVQAModel` (dual-image difference VQA).
  - [x] Expose `/api/v1/change` with change-map overlay and explanation.
  - [x] Verify change logic using mock temporal pairs.

## Upcoming Milestones

### [x] Phase 4: Cross-Modal Optical + SAR Analysis

- [x] Implement optical-SAR coregistration validator.
- [x] Implement `OpticalSARFusionModel` for joint water/built-up extraction.
- [x] Expose `/api/v1/optical-sar` route.
- [x] Verify optical+SAR combination outputs.

### [/] Phase 5: Remote-Sensing VLM Model Adaptation Layer

- [x] Select BIFOLD BigEarthNet v2.0 VQA as the adaptation dataset and document image-patch integration.
- [x] Set up Google Colab notebook for downstream fine-tuning.
- [ ] Fine-tune a model (for example, RemoteCLIP or LLaVA) on BigEarthNet or RSVQA.
- [ ] Export and load adapted checkpoints in specialist models.
- [ ] Run benchmarks comparing the adapted model with a generic VLM.

### [/] Phase 6: LangGraph Agent Orchestration

- [x] Create query task classifier router (`backend/agent/task_classifier.py`).
- [x] Define and implement an auditable deterministic flow: Validator → Router → specialist execution → evidence fusion.
- [ ] Replace the deterministic flow with a persisted LangGraph `StateGraph` when LangGraph runtime/state persistence is selected.
- [x] Expose `/api/v1/agent` returning answer, confidence, overlays, and execution trace.
- [x] Verify single-image and bi-temporal agent routing.

### [x] Phase 7: Evidence Fusion & Reports

- [x] Implement mask-overlay renderer (`backend/evidence/generator.py`).
- [x] Build PDF report export (`backend/evidence/report.py`).

### [x] Phase 8: React Web Dashboard

- [x] Develop upload UI.
- [x] Integrate Leaflet image and mask-overlay viewer.
- [x] Add interactive execution-trace dashboard.
- [x] Integrate report-download action.

### [/] Phase 9: Benchmark Evaluation

- [x] Write automated metric runners (IoU, accuracy, F1).
- [ ] Benchmark the registry against held-out evaluation subsets.
