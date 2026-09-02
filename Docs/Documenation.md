# SatQuery AI
## Complete Project Implementation and Delivery Record

**Problem statement:** SIH26167  
**Organization:** Indian Space Research Organisation (ISRO)  
**Theme:** Space Technology  
**Category:** Software  
**Document status:** Current repository record as of 2 September 2026  
**Repository branch:** `main`  
**Current revision:** `7bfb36b` (`feat: integrate BigEarthNet land cover models and VQA workflows into the backend architecture`)

---

## 1. Executive Summary

SatQuery AI is an agentic remote-sensing vision-language assistant. A user supplies one or two satellite images and a natural-language question. The system validates the files, determines the requested remote-sensing task, routes the request to a specialist model, fuses textual and spatial evidence, estimates confidence, and returns an auditable execution trace. Results can also be exported as a PDF report and displayed in a React dashboard.

The implemented system supports:

- Single-image visual question answering (VQA).
- Single-image scene captioning.
- Text-guided region grounding with bounding boxes.
- Bi-temporal change detection and change-focused VQA.
- Cross-modal optical plus SAR analysis.
- BigEarthNet v2.0 multi-label land-cover classification.
- Automatic task routing through a deterministic classifier and LangGraph `StateGraph`.
- Evidence overlays, confidence values, execution traces, and PDF reports.
- A responsive React/Leaflet web interface.
- A real BLIP LoRA remote-sensing adaptation workflow and imported checkpoint.
- Accuracy, IoU, and F1 metric runners.

The project is a working prototype and demonstration platform. Several specialist paths intentionally use deterministic image-analysis baselines so the complete demo can run without a GPU or network connection. The adapted VQA checkpoint is real, but its fixed local holdout result is only `0.50` (`5/10`) and must not be treated as a production benchmark.

---

## 2. Problem and Proposed Solution

### 2.1 Problem

Remote-sensing analysis is commonly split across isolated tools for classification, object detection, VQA, captioning, and change detection. Non-specialist users must understand satellite data, GIS workflows, model selection, sensor differences, and task-specific parameters. A generic language or vision-language model cannot be assumed to understand remote-sensing imagery reliably.

A useful system must handle:

- Optical or multispectral imagery.
- SAR imagery.
- Images acquired at different times.
- Co-registered optical/SAR pairs.
- Natural-language questions.
- Spatial evidence in addition to text.
- Domain adaptation using remote-sensing data.

### 2.2 Proposed solution

SatQuery AI acts as a natural-language controller over specialist remote-sensing tools:

1. Receive one or two image files and a query.
2. Validate existence, readability, format, dimensions, and relevant metadata.
3. Classify the intent and inspect whether the request contains a pair or cross-modal input.
4. Select the appropriate specialist model.
5. Execute the specialist workflow.
6. Combine answer text, confidence, masks, boxes, and overlays.
7. Return a structured result and an observable execution trace.
8. Optionally create a downloadable evidence report.

The system deliberately exposes the execution summary rather than internal chain-of-thought.

---

## 3. Requirements Coverage

| SIH requirement | Implemented solution | Evidence |
| --- | --- | --- |
| Remote-sensing adaptation | BLIP VQA adapted with LoRA on RSVQA data; BigEarthNet v2.0 land-cover checkpoint integrated | `experiments/adaptation/train_lora.py`, `checkpoints/rsvqa-blip-lora`, `Docs/training_runs/` |
| Single-image VQA | BLIP + LoRA model path and deterministic spectral/pixel fallback | `backend/models/vqa/model.py`, `backend/api/endpoints/vqa.py` |
| Additional single-image task | Both captioning and text-guided grounding | `backend/models/captioning/model.py`, `backend/models/grounding/model.py` |
| Multitemporal analysis | Registration checks, pixel-difference map, change VQA, temporal explanation | `backend/models/change_detection/model.py`, `backend/models/change_vqa/model.py` |
| Optical + SAR analysis | Coregistration validation and optical/backscatter fusion for water and built-up regions | `backend/models/optical_sar/model.py`, `backend/preprocessing/optical.py` |
| Agentic orchestration | Classifier, registry, LangGraph state graph, conditional routing, state/checkpoint support | `backend/agent/` |
| Evidence and auditability | Confidence, evidence, overlays, route reason, execution trace, optional PDF | `backend/evidence/`, `backend/api/endpoints/agent.py` |
| Usable prototype | React upload, result, evidence, trace, and report-download dashboard | `frontend/src/main.jsx`, `frontend/src/styles.css` |

Both preferred additional single-image capabilities were implemented: captioning and grounding.

---

## 4. Repository Structure

```text
backend/
  config.py                         Environment-backed application settings
  agent/                            Classifier, registry, state, and graph workflow
  api/                              FastAPI application and endpoint routers
  evidence/                         Mask rendering and PDF report generation
  models/                           Specialist model implementations
    base.py                         Shared specialist model contract
    captioning/
    change_detection/
    change_vqa/
    grounding/
    land_cover/
    optical_sar/
    vqa/
  preprocessing/                    Optical normalization and pair registration
  validation/                       File, image, raster, and metadata validation

datasets/
  download_samples.py               Synthetic sample generation
  fetch_real_rsvqa.py              RSVQA download/extraction utility
  rsvqa/                            Local RSVQA training and holdout data
  rsvqa_cpu80/                      Experimental 80-record CPU subset
  samples/                          Sample manifest and generated examples
  BigEarthNet.txt/                  Optional external BigEarthNet checkout

experiments/
  evaluate.py                       Metric command-line runner
  adaptation/
    train_lora.py                   BLIP LoRA adaptation entry point
    prepare_bigearthnet_manifest.py BigEarthNet streaming manifest preparation

checkpoints/
  rsvqa-blip-lora/                  Selected VQA adapter
  rsvqa-blip-lora-cpu80-1e/         Retained CPU experiment adapter

Docs/
  SatQuery_AI_SIH26167_Context.md   Problem statement and solution context
  Dataset_Integration.md            BigEarthNet VQA integration guidance
  BigEarthNet_Integration.md        BigEarthNet land-cover deployment contract
  RSVQA_VQA_Model.md                VQA adapter contract
  Execution_Checkpoints.md          Delivery checkpoints and status
  training_runs/                    Reproducible training records

frontend/
  index.html                        Vite HTML entry point
  package.json                      React/Vite/Leaflet dependencies
  src/main.jsx                      Dashboard application
  src/styles.css                    Dashboard styling

notebooks/
  remote_sensing_vlm_adaptation.ipynb Colab-oriented adaptation workflow

tests/                              Verification scripts and training tests
requirements.txt                    Python dependencies
README.md                           Quick-start and endpoint overview
TASKS.md                            Phase completion checklist
```

---

## 5. Backend Architecture

### 5.1 Application entry point

`backend/api/main.py` creates the FastAPI application. It:

- Uses `settings.API_TITLE` and `settings.API_VERSION`.
- Registers routers under `/api/v1`.
- Enables CORS with `allow_origins=["*"]`.
- Provides `GET /` for health, version, and debug status.
- Writes uploaded files to the configured upload directory using UUID-based names.
- Generally removes temporary uploads after processing.

### 5.2 Shared model boundary

`backend/models/base.py` defines the specialist model contract. Specialist implementations return structured dictionaries containing the answer and task-specific evidence rather than returning unstructured text only. This lets direct API routes and the agent use the same model layer.

### 5.3 Agent state

`backend/agent/state.py` defines `AgentState`, which carries:

- Natural-language query.
- One or more image paths.
- Requested task and image count.
- Validation status and errors.
- Selected route and route reason.
- Specialist result.
- Pair metadata.
- Execution trace.
- Optional report.
- `thread_id` for workflow state.

### 5.4 Workflow

The workflow in `backend/agent/graph.py` is:

```text
validate_inputs
      |
classify_intent
      |
select specialist execution node
      |
fuse_evidence
      |
structured response / optional report
```

The classifier in `backend/agent/task_classifier.py` is deterministic and uses both query language and image count:

- Two images plus SAR, radar, or backscatter terminology -> `optical_sar`.
- Two images otherwise -> `change`.
- Scene-description language -> `caption`.
- Locate, highlight, or bounding-box language -> `grounding`.
- Land-cover or classification language -> `land_cover`.
- Otherwise -> `vqa`.

Supported explicit `analysis_type` values are `vqa`, `caption`, `grounding`, `change`, `optical_sar`, and `land_cover`. The default is `auto`.

### 5.5 StateGraph and fallback runner

When LangGraph is available, the system builds a native `StateGraph` with conditional transitions and `MemorySaver`. If LangGraph initialization fails, an embedded DAG runner with an in-memory `_checkpoints` dictionary preserves the same node topology and behavior.

The implementation supports thread-level state handling, but persistence is process-memory based. It is not durable across process restarts. Native LangGraph history and fallback history are also separate; the current fallback history helper does not read native `MemorySaver` history.

### 5.6 Tool registry

`backend/agent/tool_registry.py` maps routes to specialists:

| Registry key | Specialist |
| --- | --- |
| `vqa` | `RemoteSensingVQAModel` |
| `caption` | `RemoteSensingCaptionModel` |
| `grounding` | `RemoteSensingGroundingModel` |
| `change_detection` | `ChangeDetectionModel` |
| `change_vqa` | `ChangeVQAModel` |
| `optical_sar` | `OpticalSARFusionModel` |
| `land_cover` | `BigEarthNetLandCoverModel` |

---

## 6. Specialist Capabilities

### 6.1 Visual Question Answering

`backend/models/vqa/model.py` implements `RemoteSensingVQAModel`.

Model-backed path:

- Base model: `Salesforce/blip-vqa-base`.
- Adapter: local `checkpoints/rsvqa-blip-lora`.
- Libraries: Hugging Face Transformers and PEFT.
- Components: `BlipProcessor` and `BlipForQuestionAnswering`.
- Lazy loading on the first request.
- Device selection: CUDA, MPS, or CPU.
- Reports `inference_mode: "model"` and `fallback_active: false`.

Fallback path:

- Uses RGB pixel heuristics when model weights are unavailable or fallback is enabled.
- Green dominance estimates vegetation.
- Blue dominance estimates water.
- Similar RGB channels with moderate brightness estimate structures or built-up areas.
- Handles keyword-oriented questions about vegetation, water, structures, descriptions, and approximate counts.

`backend/models/vqa/preprocessing.py` converts supported visualizations to the RGB contract. The BLIP adapter accepts RGB/RGBA PNG, JPEG, or TIFF visualizations. It does not accept raw 10-, 12-, or 14-band multispectral/SAR rasters.

### 6.2 Scene captioning

`backend/models/captioning/model.py` supports:

- `Salesforce/blip-image-captioning-base` model-backed captioning.
- A rule-based spectral fallback, enabled by default through `CAPTION_USE_FALLBACK=true`.

The fallback derives a scene description from vegetation, water, and gray structural ratios. This keeps captioning available in offline demonstrations.

### 6.3 Text-guided grounding

`backend/models/grounding/model.py` implements an OpenCV segmentation baseline. It:

1. Selects a target from water, vegetation, and built-up query keywords.
2. Builds a target mask.
3. Applies morphological closing.
4. Finds contours.
5. Filters small contours.
6. Returns bounding boxes in `[ymin, xmin, ymax, xmax]` format.
7. Produces a base64 annotated PNG.
8. Calculates confidence from segmented area and detected boxes.

This is not currently backed by Grounding DINO, OWL-ViT, or another learned grounding model.

### 6.4 Change detection

`backend/models/change_detection/model.py` compares a T1/T2 pair using:

- Grayscale absolute difference.
- Gaussian blur.
- Fixed threshold `30`.
- Morphological opening and closing.
- Changed-pixel counts and ratios.
- JET heatmap overlay.
- Mean green, blue, and brightness shifts.

The result includes `change_summary`, `change_ratio`, changed and total pixel counts, `change_map_b64`, confidence, evidence, and a trace.

### 6.5 Change VQA

`backend/models/change_vqa/model.py` calculates temporal changes in:

- Vegetation coverage.
- Water coverage.
- Built-up coverage.
- Overall grayscale difference.

It produces query-specific explanations for construction, vegetation, water, generic change, and related questions.

### 6.6 Optical/SAR fusion

`backend/models/optical_sar/model.py` combines complementary signals:

- Optical blue response plus low SAR backscatter indicates water.
- Neutral optical pixels plus high SAR backscatter indicates built-up areas.

`backend/preprocessing/optical.py` preprocesses SAR using median filtering, 2nd/98th percentile normalization, and conversion to 8-bit intensity.

The result includes water and built-up coverage ratios, bounding boxes, an overlay, confidence, fusion rules, and a trace.

### 6.7 BigEarthNet land-cover classification

`backend/models/land_cover/model.py` integrates the published checkpoint:

`BIFOLD-BigEarthNetv2-0/convmixer_768_32-all-v0.2.0`

The implementation:

- Loads lazily from Hugging Face.
- Requires the official reBEN repository and ConfigILM utilities.
- Reads multispectral GeoTIFFs with Rasterio.
- Uses the 12 Sentinel-2 bands `B01` through `B12`.
- Reorders, interpolates, and normalizes bands using ConfigILM preprocessing.
- Produces probabilities for 19 BigEarthNet labels.
- Returns predictions meeting `BIGEARTHNET_THRESHOLD` (default `0.5`) and all class probabilities.

The land-cover model is not an RGB classifier. The current implementation rejects RGB and raw rasters with another band count. The integration documentation describes the broader 14-band Sentinel-1/Sentinel-2 input contract (`VV`, `VH`, then `B01`-`B12`), while the current model code explicitly expects 12 Sentinel-2 bands. This contract discrepancy must be resolved before presenting 14-band inference as supported.

---

## 7. API Surface

All routes are registered under `/api/v1`.

| Method and route | Purpose |
| --- | --- |
| `POST /api/v1/vqa` | Single-image visual question answering |
| `POST /api/v1/caption` | Single-image scene captioning |
| `POST /api/v1/grounding` | Text-guided region grounding |
| `POST /api/v1/change` | Paired-image change map and question-grounded answer |
| `POST /api/v1/optical-sar` | Co-registered optical/SAR fusion |
| `POST /api/v1/land-cover` | BigEarthNet multi-label land-cover classification |
| `POST /api/v1/agent` | Automatic routing and unified response |

### 7.1 Unified agent request

`POST /api/v1/agent` accepts:

- `file_1`: required primary image.
- `file_2`: optional comparison or cross-modal image.
- `query`: natural-language request.
- `analysis_type`: optional explicit route, default `auto`.
- `include_report`: optional Boolean, default `false`.
- `thread_id`: optional state thread identifier.

A successful response can contain:

- `status`.
- `answer`.
- `confidence`.
- `route`.
- `evidence`.
- `execution_trace`.
- Task-specific artifacts such as masks, boxes, probabilities, or overlays.
- `thread_id`.
- Optional report data.

### 7.2 Validation

`backend/validation/validator.py` supports `.tif`, `.tiff`, `.png`, `.jpg`, and `.jpeg`. It checks:

- File existence and readability.
- PIL dimensions, mode, and band count.
- Rasterio metadata when available: CRS, bounds, transform, dimensions, and band count.
- Basic GeoTIFF tags when Rasterio is unavailable.

Geospatial metadata is collected but is not required for ordinary image endpoints.

`backend/preprocessing/registration.py` validates paired images for existence, readability, equal dimensions, and metadata describing spatial correspondence. Optical/SAR pairs may have different band counts, as expected, but current validation still mainly enforces readability and equal dimensions rather than proving true geospatial registration.

---

## 8. Evidence, Overlays, and Reporting

### 8.1 Overlay generation

`backend/evidence/generator.py` provides `render_mask_overlay`. It applies named masks for:

- `water`.
- `built_up`.
- `change`.

The output is a base64 PNG. Bounding boxes and specialist-generated visual artifacts can also be returned through API responses.

`generate_evidence_overlay` remains a placeholder and currently returns `overlay_generated: false`.

### 8.2 PDF reports

`backend/evidence/report.py` creates a compact PDF in memory using PIL and returns base64-encoded PDF bytes. The report includes textual fields such as:

- Status.
- Query.
- Answer.
- Confidence.
- Change summary where applicable.
- Execution trace.

The current PDF does not embed the source images or visual overlays; it is a textual evidence report.

---

## 9. React Dashboard

The frontend is a React/Vite application in `frontend/`.

Implemented workflows include:

- Primary image upload.
- Optional comparison-image upload.
- Client-side PNG/JPEG previews.
- TIFF upload with server-side preview messaging.
- Preset query buttons.
- Query textarea with a 240-character limit.
- Loading and error states.
- Route, confidence, and execution-step display.
- Expandable JSON route details.
- Leaflet evidence display for returned base64 overlays.
- Bounding-box rectangles when returned by the backend.
- PDF report download.
- Responsive desktop and mobile styling.

The dashboard always calls `/agent`; it does not expose a direct `analysis_type` selector. The backend URL is configured with `VITE_API_URL` and defaults to:

```text
http://127.0.0.1:8000/api/v1
```

Leaflet currently uses a synthetic `L.CRS.Simple` coordinate system and fixed bounds. It is an evidence viewer rather than a true geographic map.

---

## 10. Data and Dataset Work

### 10.1 Synthetic samples

`datasets/download_samples.py` creates synthetic forest and lake/suburb scenes with associated VQA and grounding metadata. These samples support deterministic tests for vegetation, water, structure, and spatial output behavior.

### 10.2 RSVQA

The repository contains:

- `datasets/rsvqa/metadata.json`.
- `datasets/rsvqa/train.jsonl`.
- `datasets/rsvqa/test_holdout.jsonl`.

The documented local training subset has 50 records. The fixed holdout contains 10 rows. `datasets/fetch_real_rsvqa.py` streams `dmarsili/RSVQA-LR-2k` from Hugging Face and writes local PNG images, metadata, and JSONL records.

`datasets/rsvqa_cpu80/` contains an 80-image/80-question extracted subset used for an experimental CPU run.

### 10.3 BigEarthNet

The selected VQA adaptation source is BIFOLD BigEarthNet v2.0. Its rows contain questions, answers, task type, category, official split, and Sentinel-1/Sentinel-2 scene identifiers. The dataset table does not directly supply the RGB image object required by the current BLIP trainer, so an image map from `s2_name` to local renderable RGB paths must be prepared. Original multispectral and SAR patches should be retained for future multimodal models.

`experiments/adaptation/prepare_bigearthnet_manifest.py`:

- Streams the Hugging Face dataset.
- Filters by the official split.
- Filters by task type.
- Maps `s2_name` to local RGB image paths.
- Writes the JSONL schema expected by `train_lora.py`.

A documented initial smoke-test command is:

```bash
python experiments/adaptation/prepare_bigearthnet_manifest.py \
  --image-map /content/s2_rgb_image_map.json \
  --output /content/bifoldearthnet_train.jsonl \
  --split train --limit 50000 --type binary
```

---

## 11. Model Adaptation and Training

### 11.1 Training entry point

`experiments/adaptation/train_lora.py` provides:

- Answer normalization.
- Deterministic train/validation splitting.
- BLIP processor-based RGB preprocessing.
- LoRA adaptation targeting BLIP `query` and `value` modules.
- Configurable epochs, batch size, learning rate, seed, validation ratio, and output directory.
- Exact normalized-answer validation.
- Adapter, processor, and `run_config.json` output.

The Colab-oriented workflow is in `notebooks/remote_sensing_vlm_adaptation.ipynb`.

### 11.2 Selected CUDA run

Record: `Docs/training_runs/2026-08-24-rsvqa-adapter.md`

- Date: 24 August 2026.
- Base model: `Salesforce/blip-vqa-base`.
- Data: `datasets/rsvqa/train.jsonl`.
- Output: `checkpoints/rsvqa-blip-lora`.
- Device: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA.
- Seed: `42`.
- Epochs: `8`.
- Recorded batch size: `1`.
- Learning rate: `5e-5`.
- Validation ratio: `0.2`.
- Maximum new tokens: `16`.
- LoRA rank: `8`.
- LoRA alpha: `16`.
- LoRA dropout: `0.05`.
- Target modules: `query`, `value`.
- Trainable parameters: `1,179,648`.
- Best validation accuracy: `0.50` (`5/10`, epoch 5).
- Average loss at epoch 8: `1.0638`.
- Status: successfully executed and promoted as the configured VQA adapter.

The training command recorded for the run was:

```bash
python experiments/adaptation/train_lora.py \
  --train-jsonl datasets/rsvqa/train.jsonl \
  --output-dir checkpoints/rsvqa-blip-lora \
  --epochs 8 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --seed 42
```

The run record notes a discrepancy between the command's batch-size argument and the recorded effective batch size. That should be clarified in a future reproducibility pass.

### 11.3 CPU experiment

Record: `Docs/training_runs/2026-08-25-rsvqa-augmented-cpu.md`

- Source: `dmarsili/RSVQA-LR-2k` cached validation split.
- Data: 80 real satellite images and 80 question-answer records.
- Manifest: `datasets/rsvqa_cpu80/train.jsonl`.
- Base model: `Salesforce/blip-vqa-base`.
- LoRA rank: `8`; query/value target modules.
- Device: CPU, `torch 2.13.0+cpu`.
- Epochs: `1`.
- Batch size: `1`.
- Learning rate: `5e-5`.
- Validation split: `20%`, 16 records.
- Checkpoint: `checkpoints/rsvqa-blip-lora-cpu80-1e`.
- Validation accuracy: `0.1875` (`3/16`).
- Average training loss: `1.7726`.
- Status: retained as an experiment and not promoted.

This result does not prove the augmented adapter is worse. A meaningful comparison requires a fixed, leakage-free holdout and more training on GPU or cloud hardware.

### 11.4 Runtime model contract

The selected adapter is loaded over `Salesforce/blip-vqa-base`. The repository stores the adapter and processor artifacts, but not the full BLIP base weights. With local-files-only mode enabled, the base model must already be cached. The API exposes whether each answer came from the model or fallback path.

---

## 12. Evaluation

`backend/evaluation/metrics.py` implements:

- Exact classification accuracy.
- Binary IoU.
- Binary F1.

`experiments/evaluate.py` reads JSONL rows and optionally calculates:

- Accuracy from `prediction` and `target`.
- Mean IoU from `pred_mask` and `true_mask`.
- Mean F1 from `pred_mask` and `true_mask`.

Current status:

- The metric runner executes successfully.
- Metric primitives are tested.
- The available 10-row manifest contains earlier predictions and is not a representative registry benchmark.
- An adapted-versus-generic VLM comparison remains pending.
- The selected VQA score of `0.50` is prototype evidence on a small fixed holdout, not a production claim.

---

## 13. Verification and Tests

| Test or script | Coverage |
| --- | --- |
| `tests/verify_vqa.py` | Root endpoint and synthetic green/blue VQA behavior |
| `tests/verify_phase2.py` | Captioning and grounding |
| `tests/verify_phase3.py` | Temporal change detection |
| `tests/verify_phase4.py` | Optical/SAR fusion |
| `tests/verify_agent.py` | Agent routing, reports, VQA, grounding, optical/SAR, and checkpoints |
| `tests/verify_model_vqa.py` | Model-backed BLIP/LoRA smoke test |
| `tests/test_training_utils.py` | Answer normalization and deterministic splitting |
| `tests/verify_evaluation.py` | Evaluation metric primitives |
| `tests/verify_phase2.py` through `verify_phase4.py` | Phase-specific specialist verification |

The model-backed smoke test requires cached BLIP base weights and checks both direct `/api/v1/vqa` and automatic VQA selection through `/api/v1/agent`. Both must report `inference_mode: model`.

Known test caveats:

- `tests/verify_agent.py` can fail its history assertion when native LangGraph is active because its `get_state_history()` check reads the fallback runner's dictionary rather than native `MemorySaver` state.
- `tests/verify_evaluation.py` is not directly executable unless the project root is added to `sys.path`.
- The frontend production build was not verified in the recorded environment because Node/npm dependencies were unavailable.

---

## 14. Configuration

`backend/config.py` loads settings from `.env` through Pydantic Settings. Unknown variables are ignored. The upload directory is created at import time.

| Setting | Default or purpose |
| --- | --- |
| `API_TITLE` | `SatQuery AI API` |
| `API_VERSION` | `v1` |
| `DEBUG` | `true` |
| `VQA_MODEL_NAME` | BLIP VQA base model identifier |
| `VQA_USE_FALLBACK` | `false`; set `true` for deterministic pixel fallback |
| `VQA_LOCAL_FILES_ONLY` | `true`; prevents automatic base-model download |
| `VQA_MAX_NEW_TOKENS` | `16` |
| `VQA_NUM_BEAMS` | `4` |
| `CAPTION_USE_FALLBACK` | `true` |
| `VQA_ADAPTER_PATH` | Local VQA adapter directory |
| `BIGEARTHNET_MODEL_ID` | BigEarthNet ConvMixer checkpoint |
| `BIGEARTHNET_EXPECTED_BANDS` | `12` in current implementation |
| `BIGEARTHNET_THRESHOLD` | `0.5` |
| `UPLOAD_DIR` | Temporary upload location |
| `VITE_API_URL` | Frontend API base URL, default `http://127.0.0.1:8000/api/v1` |

---

## 15. Installation and Operation

### 15.1 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, use the equivalent `.venv\Scripts\Activate.ps1` activation command.

### 15.2 Start the API

```bash
python3 -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 15.3 BigEarthNet prerequisite

The BigEarthNet model uses the custom model class from the official reBEN repository:

```bash
pip install -r requirements.txt
git clone https://git.tu-berlin.de/rsim/reben-training-scripts.git external/reben
export PYTHONPATH="$PWD/external/reben:$PYTHONPATH"
```

On Windows PowerShell, the Python path equivalent is:

```powershell
$env:PYTHONPATH = "$PWD\external\reben;$env:PYTHONPATH"
```

The integration imports `reben_publication.BigEarthNetv2_0_ImageClassifier`. On first inference, the checkpoint downloads from Hugging Face unless already cached. BigEarthNet inference also requires Rasterio, ConfigILM, Hugging Face access or cache, and the official reBEN model class.

### 15.4 VQA model-backed smoke test

After caching the BLIP base model:

```bash
.venv/bin/python tests/verify_model_vqa.py
```

The fallback route can be deliberately selected for offline deterministic demonstrations with:

```text
VQA_USE_FALLBACK=true
```

### 15.5 Frontend

Install the frontend dependencies from `frontend/package.json`, configure `VITE_API_URL` when the API is not using its default address, and run the Vite development workflow. The recorded repository checkpoint did not include a verified production build because Node/npm dependencies were unavailable in that environment.

---

## 16. Delivery Milestones and Checkpoints

### Phase 0: Project structure

Completed. The backend, models, API, preprocessing, validation, evidence, datasets, experiments, notebooks, tests, and frontend layout were created. Base configuration and dependency declarations were added.

### Phase 1: Foundation and single-image VQA

Completed. The shared model base, validator, VQA specialist, FastAPI routes, deterministic fallback, and green/blue synthetic-image tests were implemented.

### Phase 2: Captioning and grounding

Completed. BLIP/rule-based captioning, OpenCV grounding, API routes, and coordinate verification were added.

### Phase 3: Bi-temporal change analysis

Completed. Pair compatibility checks, difference-map generation, change VQA, change overlays, and temporal mock-pair verification were added.

### Phase 4: Optical/SAR analysis

Completed. Optical-SAR validation, preprocessing, fusion rules, route exposure, and verification were added.

### Phase 5: VLM adaptation

Implemented, with benchmark work still open. BigEarthNet/RSVQA data preparation, Colab workflow, LoRA training, real CUDA adapter, adapter loading, and training records exist. A representative adapted-versus-generic benchmark is pending.

### Phase 6: LangGraph orchestration

Completed. The classifier, registry, validator-router-specialist-fusion flow, native StateGraph, conditional edges, thread state, fallback runner, agent route, and routing verification exist.

### Phase 7: Evidence and reports

Completed at prototype level. Mask overlays and PDF export exist. The PDF is textual and the generic evidence overlay entry point remains incomplete.

### Phase 8: React dashboard

Implemented. Uploads, paired images, results, trace, evidence display, and report export are present. A production build still requires Node/npm dependencies and verification.

### Phase 9: Benchmark evaluation

Partially complete. Accuracy, IoU, and F1 runners exist and execute, but the available manifest is not representative and the registry benchmark plus generic-model comparison remain open.

### Checkpoint schedule

| Checkpoint | Target | Status |
| --- | --- | --- |
| C1 | 24 Aug: adaptation workflow | Complete |
| C2 | 25 Aug: agent orchestration | Complete |
| C3 | 26 Aug: evidence export | Complete |
| C4 | 27 Aug: evaluation runner | Partially complete |
| C5 | 28 Aug: dashboard | Implemented; build verification pending |
| C6 | 29-31 Aug: integration rehearsal | Partially complete; larger real-data rehearsal pending |

---

## 17. Known Limitations and Risks

1. **Small VQA validation set:** The promoted adapter's `0.50` score is only `5/10` on a fixed local holdout.
2. **No generic comparison:** Adapted-versus-generic VLM benchmarking is still pending.
3. **Non-representative registry benchmark:** The current 10-row evaluation manifest contains earlier predictions.
4. **Base-model dependency:** The repository has the LoRA adapter but requires cached BLIP base weights.
5. **Fallback baselines:** Captioning, grounding, change detection, and optical/SAR are deterministic prototype baselines rather than trained specialist models.
6. **VQA input limitation:** The BLIP adapter accepts RGB visualizations, not raw multispectral or SAR rasters.
7. **BigEarthNet dependency chain:** Land-cover inference requires Rasterio, ConfigILM, Hugging Face access/cache, and the official reBEN code.
8. **BigEarthNet validation gap:** End-to-end inference on an official co-registered Sentinel-1/Sentinel-2 GeoTIFF has not been validated.
9. **Band-contract discrepancy:** Documentation describes a 14-band Sentinel-1/Sentinel-2 input, but current code expects 12 Sentinel-2 bands.
10. **Pair registration depth:** Pair validation does not yet prove CRS equality, resolution equality, transform compatibility, geographic overlap, or true coregistration.
11. **Internal resizing:** Change models resize mismatched images internally, although API validation rejects mismatched dimensions first.
12. **Evidence report scope:** PDFs do not embed source images or overlays.
13. **Placeholder evidence function:** `generate_evidence_overlay` is not fully implemented.
14. **Synthetic map coordinates:** Leaflet uses fixed `L.CRS.Simple` bounds rather than geographic coordinates.
15. **Process-memory persistence:** Agent state is not durable across process restarts.
16. **Native history mismatch:** The history helper does not currently unify fallback and native LangGraph checkpoint history.
17. **Frontend build verification:** Node/npm dependencies were unavailable in the recorded environment.
18. **Training record discrepancy:** The selected run records batch size `1`, while its command specifies `--batch-size 2`; reproducibility documentation should clarify the effective value.

---

## 18. Recommended Next Work

1. Resolve and test one authoritative BigEarthNet band contract: either the current 12-band Sentinel-2 classifier or a complete 14-band Sentinel-1/Sentinel-2 path.
2. Clone and pin the reBEN dependency, then run end-to-end inference on an official compatible GeoTIFF.
3. Build a leakage-free, representative evaluation split for every registry specialist.
4. Compare the adapted VQA model against the generic BLIP baseline using identical data and metrics.
5. Unify native LangGraph and fallback history access and add a regression test for both paths.
6. Strengthen pair validation with CRS, transform, resolution, overlap, and registration checks.
7. Embed overlays and source-image evidence in PDF reports.
8. Replace or complement deterministic grounding and captioning baselines with domain-adapted learned models where resources permit.
9. Install frontend dependencies and run a production build and browser verification.
10. Clarify the recorded VQA batch-size discrepancy and record checkpoint checksums for complete reproducibility.

---

## 19. Change History

The repository history shows the following major completed increments:

| Revision | Contribution |
| --- | --- |
| `9c1b1e6` | Introduced LangGraph StateGraph orchestration layer |
| `7b43a81` | Added GPU fine-tuned adapter and training artifacts |
| `f0011ca` | Added RSVQA LoRA checkpoint and training record |
| `708ba78` | Improved RSVQA adapter training and benchmark record |
| `2ecbdc8` | Improved LoRA pipeline and recorded validation accuracy |
| `3e6c58f` | Fixed validated LoRA checkpoint workflow |
| `4162fd3` | Improved VQA adapter validation accuracy |
| `59fa73e` | Reconciled project task completion status |
| `21639a1` | Improved satellite analysis frontend |
| `c6aad1d` | Fixed frontend |
| `9dfe4c9` | Completed SatQuery MVP integration |
| `4622976` | Enabled the trained VQA adapter by default |
| `7bfb36b` | Integrated BigEarthNet land-cover models and VQA workflows into backend architecture |

The working checkout was force-synchronized with `origin/main` before this documentation was created. The final worktree was clean before adding this document.

---

## 20. Primary References

- `README.md` - Quick-start, project structure, and API overview.
- `TASKS.md` - Phase checklist and current milestone status.
- `Docs/SatQuery_AI_SIH26167_Context.md` - Official problem context and proposed approach.
- `Docs/Dataset_Integration.md` - BigEarthNet VQA dataset preparation.
- `Docs/BigEarthNet_Integration.md` - BigEarthNet land-cover model contract.
- `Docs/RSVQA_VQA_Model.md` - BLIP LoRA runtime and input contract.
- `Docs/Execution_Checkpoints.md` - Delivery schedule, rules, and checkpoint status.
- `Docs/training_runs/2026-08-24-rsvqa-adapter.md` - Selected CUDA training run.
- `Docs/training_runs/2026-08-25-rsvqa-augmented-cpu.md` - Retained CPU experiment.