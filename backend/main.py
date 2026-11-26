from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database import init_db
from backend.routers import auth, applications, documents, predictions, officer, fairness


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.DEBUG and settings.SECRET_KEY.startswith("creditxplain-secret-key"):
        raise RuntimeError("Set SECRET_KEY in .env before running with DEBUG=False")
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Explainable Credit Risk Prediction API with Counterfactual Recommendations",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(applications.router, prefix="/api/applications", tags=["Applications"])
app.include_router(documents.router, prefix="/api/applications", tags=["Documents"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(officer.router, prefix="/api/officer", tags=["Officer Dashboard"])
app.include_router(fairness.router, prefix="/api/fairness", tags=["Fairness Monitoring"])

@app.get("/")
def root():
    return {
        "message": "CreditXplain API",
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
