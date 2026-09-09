"""FastAPI application entry point."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI

from routers import sessions

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

app.include_router(sessions.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
