import os
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "SatQuery AI API"
    API_VERSION: str = "v1"
    DEBUG: bool = True
    
    # Model Configuration
    VQA_MODEL_NAME: str = "Salesforce/blip-vqa-base"
    VQA_USE_FALLBACK: bool = True
    VQA_ADAPTER_PATH: str | None = None
    
    # Paths
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

    @field_validator("DEBUG", mode="before")
    @classmethod
    def normalize_debug_value(cls, value):
        """Accept common deployment-mode values used by local `.env` files."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure uploads directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
