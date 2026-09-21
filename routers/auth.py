"""Auth routes: account creation and sign-in.

Scope of this module is #86 — register + login. The sign-in cookie is a
30-day JWT (httpOnly, SameSite=Lax) set by the backend on the shared
``localhost`` host so both the frontend origin and the API can see it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session as DBSession

from core import security
from db import User, get_db

router = APIRouter(tags=["auth"])

MIN_PASSWORD_LENGTH = 8


# --- Schemas ---


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    # Optional; when omitted/blank the display name defaults to the local
    # part of the email (e.g. "ada" for "ada@example.com").
    display_name: str | None = None

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        if len(value) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        return value


class LoginRequest(BaseModel):
    # No length validation here: a short password at login is just an
    # invalid credential (401), never a 422 that would tip off enumeration.
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str


class Message(BaseModel):
    message: str


# --- Routes ---


@router.post("/auth/register", response_model=UserOut, status_code=201)
def register(payload: RegisterRequest, db: DBSession = Depends(get_db)) -> UserOut:
    """Create an account. Duplicate email → 409, weak password → 422."""
    email = payload.email.lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already in use")

    display_name = payload.display_name or email.split("@", 1)[0]
    user = User(
        email=email,
        display_name=display_name,
        password_hash=security.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, email=user.email, display_name=user.display_name)


@router.post("/auth/login", response_model=Message)
def login(
    payload: LoginRequest,
    response: Response,
    db: DBSession = Depends(get_db),
) -> Message:
    """Sign in with email + password. Invalid credentials → generic 401."""
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not security.verify_password(
        user.password_hash, payload.password
    ):
        # Generic message so a wrong password is indistinguishable from an
        # unknown email (no account enumeration).
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = security.create_token(user.email)
    response.set_cookie(
        security.COOKIE_NAME,
        token,
        max_age=security.COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return Message(message="Signed in")
