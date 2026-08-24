from backend.api.endpoints.vqa import router as vqa_router
from backend.api.endpoints.caption import router as caption_router
from backend.api.endpoints.grounding import router as grounding_router
from backend.api.endpoints.change import router as change_router
from backend.api.endpoints.optical_sar import router as optical_sar_router
from backend.api.endpoints.agent import router as agent_router

__all__ = ["vqa_router", "caption_router", "grounding_router", "change_router", "optical_sar_router", "agent_router"]
