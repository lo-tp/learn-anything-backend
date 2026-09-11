"""Internal slide content route: raw HTML fetch by slide ID.

Not public — only the **sandbox service** calls this to fetch the raw slide
HTML it renders/serves to the client. The client never calls this endpoint
directly (see docs/api-and-graphs.md).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import SlideContent, get_db

router = APIRouter(tags=["slides"])


class SlideOut(BaseModel):
    slide_id: str
    content: str


@router.get("/slides/{slide_id}", response_model=SlideOut)
def get_slide(slide_id: str, db: DBSession = Depends(get_db)) -> SlideOut:
    """Fetch raw slide HTML by globally-unique slide ID (internal)."""
    slide = db.get(SlideContent, slide_id)
    if slide is None:
        raise HTTPException(status_code=404, detail="Slide not found")
    return SlideOut(slide_id=slide.slide_id, content=slide.content)
