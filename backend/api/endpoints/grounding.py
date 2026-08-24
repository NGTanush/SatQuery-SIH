import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.config import settings
from backend.validation.validator import InputValidator
from backend.agent.tool_registry import tool_registry

logger = logging.getLogger("satquery.api.grounding")
router = APIRouter()

@router.post("/grounding", status_code=status.HTTP_200_OK)
async def execute_grounding(
    file: UploadFile = File(..., description="Satellite image (GeoTIFF/TIFF, PNG, JPEG)"),
    query: str = Form(..., description="Target class query (e.g. 'water body', 'vegetation', 'built-up structures')")
):
    """
    Upload a satellite image and localize regions matching the query.
    Returns coordinates of bounding boxes and a highlighted base64 annotated image.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query target cannot be empty."
        )

    file_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1]
    ext = original_ext if original_ext else ".tiff"
    
    file_name = f"{file_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)

    try:
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write uploaded file: {str(e)}"
        )

    # Validate image file
    is_valid, error_msg, metadata = InputValidator.validate_image(file_path)
    if not is_valid:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {error_msg}"
        )

    # Fetch tool from registry
    grounding_tool = tool_registry.get_tool("grounding")
    if not grounding_tool:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounding specialist tool is not registered in the system registry."
        )

    try:
        # Run tool
        result = grounding_tool.run({
            "image_path": file_path,
            "query": query
        })
        
        # Inject metadata
        result["evidence"]["image_metadata"] = metadata
        result["query"] = query
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        logger.error(f"Error during Grounding execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounding execution failed: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temporary file {file_path}: {str(cleanup_err)}")
