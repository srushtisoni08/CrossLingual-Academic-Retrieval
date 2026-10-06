import logging
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import BACKEND_DIR, EMBEDDING_MODEL
from app.routes import papers, search
from app.services import retrieval

load_dotenv(BACKEND_DIR / ".env")
logging.basicConfig(level=logging.INFO)

# Comma-separated list in backend/.env, e.g. CORS_ORIGINS=http://localhost:5173
ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieval.warm_up()  # builds any missing index, loads model once
    threading.Thread(target=retrieval.warm_up_optional, daemon=True).start()
    yield


app = FastAPI(
    title="CrossLingual Academic Retrieval API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ORIGINS],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api")
app.include_router(papers.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": EMBEDDING_MODEL}