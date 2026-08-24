from backend.api.endpoints.vqa import router as vqa_router
from backend.api.endpoints.caption import router as caption_router
from backend.api.endpoints.grounding import router as grounding_router

__all__ = ["vqa_router", "caption_router", "grounding_router"]
