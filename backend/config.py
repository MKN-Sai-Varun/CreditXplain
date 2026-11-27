from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=True)
    
    # Application
    APP_NAME: str = "CreditXplain API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "creditxplain-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database
    DATABASE_URL: str = "sqlite:///./creditxplain.db"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data" / "processed"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # Model
    MODEL_PATH: Path = MODELS_DIR / "final_lgbm.pkl"
    SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
    CONTINUOUS_COLS_PATH: Path = MODELS_DIR / "continuous_cols.pkl"
    
    # Decision thresholds (from Phase 3)
    APPROVE_THRESHOLD: float = 0.0349
    REJECT_THRESHOLD: float = 0.1189
    
    # Fairness thresholds
    FAIRNESS_DPD_THRESHOLD: float = 0.1
    
    # Loan calculation
    BASE_INTEREST_RATE: float = 0.10  # 10% base rate
    RISK_RATE_SPREAD: dict = {
        "APPROVE": 0.02,   # +2% (12% total)
        "REVIEW": 0.05,    # +5% (15% total)
        "REJECT": 0.08     # +8% (18% total, for manual override cases)
    }
    
    # Loan tenure options (months)
    TENURE_OPTIONS: list = [12, 24, 36, 48, 60]
    DEFAULT_TENURE_MONTHS: int = 12

    # Recourse preferences an applicant can set
    RECOURSE_LEVERS: list = ["credit", "income", "tenure", "wait"]
    WAIT_OPTIONS: list = [0, 3, 6, 12]
    MAX_LOAN_REDUCTION: float = 0.5

    # List options that survive retrained models first (bootstrap models 1-5, if present)
    ROBUST_RECOURSE: bool = True

    # Document uploads
    MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024
    ALLOWED_UPLOAD_TYPES: dict = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]

settings = Settings()
