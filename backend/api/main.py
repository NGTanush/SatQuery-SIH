import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api.endpoints import vqa_router, caption_router, grounding_router

# Configure Logger
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("satquery.api")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Backend API services for SatQuery AI - Interactive Remote Sensing Vision-Language Assistant."
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(vqa_router, prefix="/api/v1", tags=["Specialist Tools"])
app.include_router(caption_router, prefix="/api/v1", tags=["Specialist Tools"])
app.include_router(grounding_router, prefix="/api/v1", tags=["Specialist Tools"])

@app.get("/")
async def root():
    return {
        "app": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "healthy",
        "debug_mode": settings.DEBUG
    }
