"""FastAPI application entry point."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import clarify, plan, probe, sessions

load_dotenv()

# Log level is controlled by LOG_LEVEL in .env (default: INFO).
# Set LOG_LEVEL=DEBUG for per-step / per-node material generation detail.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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

app.include_router(sessions.router)
app.include_router(clarify.router)
app.include_router(probe.router)
app.include_router(plan.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
