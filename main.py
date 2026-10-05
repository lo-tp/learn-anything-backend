"""FastAPI application entry point."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from core.mock_llm import is_mock_mode
from db import seed_mock_user
from routers import auth, clarify, plan, probe, review, sessions, slides

# Log level is controlled by LOG_LEVEL in .env (default: INFO).
# Set LOG_LEVEL=DEBUG for per-step / per-node material generation detail.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if is_mock_mode():
    logging.getLogger(__name__).warning(
        "RUNNING IN MOCK MODE: pre-material LLM mocked, slide generation "
        "disabled, in-memory DB (seeded with a fixed fake user)."
    )
    # Feed a fixed fake account into the RAM DB's users table before the app
    # starts serving, so a known login is always available in mock mode.
    seed_mock_user()

PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

FRONTEND_DOMAIN = os.getenv("FRONTEND_DOMAIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_DOMAIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(clarify.router)
app.include_router(probe.router)
app.include_router(plan.router)
app.include_router(slides.router)
app.include_router(review.router)


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    """Liveness probe for the platform health check (Render polls this).

    Deliberately cheap and dependency-free: it answers as soon as the app can
    serve a request, so a slow or unreachable Postgres shows up as request
    errors in the logs rather than a restart loop.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        timeout_graceful_shutdown=5,
    )
