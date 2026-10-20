"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from core.mock_llm import is_mock_mode
from core.settings import CORS_ORIGINS_ENV, origin_list_setting
from db import seed_mock_user
from routers import auth, clarify, explore, plan, probe, review, sessions, slides

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start serving with no work in flight.

    A regeneration mark means "this process is rewriting this slide", and a
    process that has just started is rewriting nothing — so every mark in the
    database was left by one that died. Clearing them is what keeps the deck
    honest and keeps a slide regenerable (a marked slide refuses new requests).
    """
    plan.clear_abandoned_regenerations()
    yield


app = FastAPI(lifespan=lifespan)

_DEFAULT_FRONTEND_DOMAIN = "http://localhost:3000"


def cors_allow_origins() -> list[str]:
    r"""Every origin this API will answer, in the order it will match them.

    ``FRONTEND_DOMAIN`` comes first, because it is the origin the deployment is
    actually served from and the one every existing environment already names —
    including the infrastructure repo's prod overlay, which sets
    ``https://learn.lotp.xyz``. It is not replaced by ``CORS_ORIGINS``; a
    setting that adds origins is what a developer debugging from another device
    needs, and a setting that swaps the served origin out is how a deployment
    quietly stops being able to sign itself in.

    Duplicates are removed keeping the first, so writing the deployment origin
    into ``CORS_ORIGINS`` as well (the intuitive thing to do when a request from
    it is failing) changes nothing instead of producing a list with two answers
    for one origin.

    Read per call rather than at import, like :func:`core.security.cookie_policy`,
    so the list a test builds is the list the app would build.
    """
    served = os.getenv("FRONTEND_DOMAIN", _DEFAULT_FRONTEND_DOMAIN).strip()
    requested = origin_list_setting((CORS_ORIGINS_ENV,))
    origins: list[str] = []
    seen: set[str] = set()
    for origin in ([served] if served else []) + requested:
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


_ALLOWED_ORIGINS = cors_allow_origins()
# Named once, at startup, because the failure this prevents is a browser refusing
# a request with a message that points at the browser. A developer whose phone
# gets a CORS error should be able to see from the log which origins this process
# will answer, without reading its environment file.
logging.getLogger(__name__).info(
    "CORS: answering %d origin(s): %s",
    len(_ALLOWED_ORIGINS),
    ", ".join(_ALLOWED_ORIGINS) or "none (no origin is allowed)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(explore.router)
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
