import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "SatQuery AI API"
    API_VERSION: str = "v1"
    DEBUG: bool = True
    
    # Model Configuration
    VQA_MODEL_NAME: str = "Salesforce/blip-vqa-base"
    VQA_USE_FALLBACK: bool = True
    
    # Paths
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure uploads directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
