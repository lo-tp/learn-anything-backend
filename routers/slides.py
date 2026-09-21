"""Internal slide content route: raw JSX fetch by slide ID.

Not public — only the **sandbox service** calls this to fetch the raw slide
JSX (React component source) it compiles and mounts to serve the client. The
client never calls this endpoint directly (see docs/api-and-graphs.md).

Auth is the service-identity gate (``require_service``), not the human
cookie gate: a valid ``X-Service-Token`` header is required, and
``DEV_MODE`` never opens it.
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from core.security import require_service
from db import SlideContent, get_db

router = APIRouter(tags=["slides"], dependencies=[Depends(require_service)])


class SlideOut(BaseModel):
    slide_id: str
    content: str


@router.get("/slides/{slide_id}", response_model=SlideOut)
def get_slide(slide_id: str, db: DBSession = Depends(get_db)) -> SlideOut:
    """Fetch raw slide JSX (component source) by globally-unique slide ID (internal)."""
    slide = db.get(SlideContent, slide_id)
    # Placeholder slides (a failed slide's stand-in) are only served in dev
    # mode; outside dev they are invisible, as if the slide never existed.
    if slide is None or (slide.is_placeholder and os.getenv("DEV_MODE") != "1"):
        raise HTTPException(status_code=404, detail="Slide not found")
    return SlideOut(slide_id=slide.slide_id, content=slide.content)
