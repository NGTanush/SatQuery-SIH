import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.agent.task_classifier import TaskClassifier
from backend.agent.tool_registry import tool_registry
from backend.config import settings
from backend.evidence.report import generate_pdf_report
from backend.preprocessing.registration import ImageRegistration
from backend.validation.validator import InputValidator

logger = logging.getLogger("satquery.api.agent")
router = APIRouter()
classifier = TaskClassifier()


async def _persist(file: UploadFile, index: int) -> str:
    extension = os.path.splitext(file.filename or "")[1] or ".tiff"
    path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}_agent_{index}{extension}")
    with open(path, "wb") as output:
        while part := await file.read(1024 * 1024):
            output.write(part)
    return path


@router.post("/agent", status_code=status.HTTP_200_OK)
async def execute_agent(
    file_1: UploadFile = File(..., description="Primary image"),
    file_2: UploadFile | None = File(None, description="Optional paired image"),
    query: str = Form(...),
    analysis_type: str = Form("auto"),
    include_report: bool = Form(False),
):
    """Validate input, route to a specialist, and return auditable evidence."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    paths = []
    try:
        first = await _persist(file_1, 1)
        paths.append(first)
        second = None
        if file_2:
            second = await _persist(file_2, 2)
            paths.append(second)
        for path in paths:
            valid, error, _ = InputValidator.validate_image(path)
            if not valid:
                raise HTTPException(status_code=400, detail=f"Image validation failed: {error}")
        decision = classifier.classify(query, len(paths), analysis_type)
        if decision.task in {"change", "optical_sar"} and not second:
            raise HTTPException(status_code=400, detail=f"{decision.task} analysis requires two images.")
        if second:
            valid, error, pair_metadata = ImageRegistration.validate_pair(first, second)
            if not valid:
                raise HTTPException(status_code=400, detail=f"Pair validation failed: {error}")
        else:
            pair_metadata = None
        tool = tool_registry.get_tool(decision.task if decision.task not in {"change", "optical_sar"} else decision.task)
        if decision.task == "change":
            detection = tool_registry.get_tool("change_detection")
            change_vqa = tool_registry.get_tool("change_vqa")
            if not detection or not change_vqa:
                raise HTTPException(status_code=503, detail="Change tools are unavailable.")
            model_inputs = {"image_path_a": first, "image_path_b": second}
            detected, answered = detection.run(model_inputs), change_vqa.run({**model_inputs, "question": query})
            result = {
                "answer": answered["answer"],
                "confidence": answered["confidence"],
                "change_summary": detected["change_summary"],
                "overlay_b64": detected["change_map_b64"],
                "evidence": {"change_detection": detected["evidence"], "change_vqa": answered["evidence"]},
                "execution_trace": {"steps": [detected["execution_trace"], answered["execution_trace"]]},
            }
        elif decision.task == "optical_sar":
            if not tool:
                raise HTTPException(status_code=503, detail="Optical-SAR fusion tool is unavailable.")
            result = tool.run({"optical_path": first, "sar_path": second, "query": query})
            result["answer"] = result["summary"]
        else:
            if not tool:
                raise HTTPException(status_code=503, detail=f"{decision.task} tool is unavailable.")
            params = {"image_path": first}
            if decision.task == "vqa":
                params["question"] = query
            elif decision.task == "grounding":
                params["query"] = query
            result = tool.run(params)
            result["answer"] = result.get("answer") or result.get("caption") or f"Detected {result.get('box_count', 0)} regions."
        result.update({"status": "success", "query": query, "route": {"task": decision.task, "reason": decision.reason}, "pair_metadata": pair_metadata})
        if include_report:
            result["report_pdf_b64"] = generate_pdf_report("SatQuery AI Evidence Report", result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc
    finally:
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
