"""Password hashing and sign-in token (JWT) helpers.

The sign-in state is a stateless 30-day JWT carried in an httpOnly cookie.
There is no server-side revocation: logout only clears the presenting
client's cookie (handled elsewhere), so a token stays valid until it expires.
"""

from __future__ import annotations

import hmac
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import Depends, HTTPException, Request

from db import get_db

if TYPE_CHECKING:
    from db import User

_JWT_ALGORITHM = "HS256"

# 30-day sign-in state.
TOKEN_TTL = timedelta(days=30)
COOKIE_MAX_AGE_SECONDS = int(TOKEN_TTL.total_seconds())

COOKIE_NAME = "access_token"


def cookie_policy() -> dict:
    """The sign-in cookie's scope, spelled once.

    Two services sit on one domain: ``api.`` issues this cookie and ``learn.``
    reads it (``proxy.ts`` verifies it with the same ``JWT_SECRET``). A cookie
    written without a ``domain`` is host-only, so the app's own gate can never
    see that a user has signed in — it would bounce a signed-in browser back to
    the login page forever. Setting ``COOKIE_DOMAIN`` to the shared parent
    (``.lotp.xyz``) is what makes the two halves one session; leaving it unset
    keeps the cookie host-only, which is what a developer running both services
    on localhost wants.

    ``Secure`` follows the frontend origin's scheme rather than adding a flag of
    its own: one setting already says whether this deployment is served over
    https, and two flags that can disagree is one too many.
    """
    return {
        "domain": os.getenv("COOKIE_DOMAIN") or None,
        "secure": os.getenv("FRONTEND_DOMAIN", "").startswith("https://"),
        "httponly": True,
        "samesite": "lax",
        "path": "/",
    }


def _jwt_secret() -> str:
    """The shared signing secret (parent #85).

    Read per call so a late env change (tests toggling it, a reloaded
    ``.env``) is honoured; called once at import to fail fast at startup
    if it is absent, mirroring ``DATABASE_URL`` in ``db.models``.
    """
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET is not set. "
            "Expected the shared secret used to sign/verify the sign-in token."
        )
    return secret


# Validate presence now (fail fast); the value is still read per call.
_jwt_secret()


# One hasher instance is reused: argon2-cffi precomputes its salt length and
# parameters once, which is what the docs recommend for repeated use.
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Return an argon2 hash string for ``password``."""
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Return True if ``password`` matches ``password_hash``.

    Never raises: a corrupt or non-argon2 stored value is reported as a
    mismatch rather than an error, so login fails as a generic 401.
    """
    try:
        return _hasher.verify(password_hash, password)
    except VerificationError:
        return False


def create_token(email: str) -> str:
    """Encode a 30-day HS256 JWT whose subject is the user's email."""
    now = datetime.now(UTC)
    payload = {
        "sub": email,
        "iat": now,
        "exp": now + TOKEN_TTL,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_JWT_ALGORITHM)


# --- Sign-in gates for learning-session endpoints (#89, #145) ---


def is_dev_mode() -> bool:
    """The shared DEV_MODE predicate (secure default: off).

    Read per request so tests (and runtime toggles) take effect
    immediately. Any endpoint that keys off dev mode must use this
    helper so the flag has one meaning application-wide.
    """
    return os.getenv("DEV_MODE") in ("true", "1")


def _decode_token(request: Request) -> dict:
    """Extract and validate the sign-in token; 401 on missing/invalid."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[_JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_sign_in(request: Request) -> None:
    """FastAPI dependency: require a valid sign-in token, always.

    Never opened by ``DEV_MODE``. This is the gate for the one endpoint that
    brings a Session into existence — ``POST /sessions``, #145. A Session with
    no one who asked for it becomes impossible: the refusal lands before the
    write. The answer is the same 401 as the bounded gate's, so the client has
    one sign-in signal to react to (#143: the 401 *is* the identity signal).

    The Session **reads** use neither gate: the list, one Session's state and
    its materials are public (#178), and ownership begins at the write.
    """
    _decode_token(request)


def require_auth(request: Request) -> None:
    """FastAPI dependency: require a valid sign-in token unless DEV_MODE.

    - ``DEV_MODE`` is read from the environment on **every request** so
      tests (and runtime toggles) take effect immediately.
    - Missing or invalid cookie → 401.
    - When the gate passes, no per-user scoping is applied: sessions
      remain shared across all users.

    The Session **write** endpoints use this — the intake phases (clarify,
    probe) and the plan writes — so a local run can walk the phases without a
    cookie. It is declared per route where a router also serves a public read
    (``routers.plan``, #178). Anything that creates data uses
    :func:`require_sign_in`, whatever the flag says.
    """
    if is_dev_mode():
        return
    require_sign_in(request)


# --- Service-identity gate for the internal slides endpoint (#103) ---

SERVICE_TOKEN_HEADER = "X-Service-Token"


def require_service(request: Request) -> None:
    """FastAPI dependency: require the shared Sandbox service token.

    Fail-secure: ``SANDBOX_SERVICE_TOKEN`` is read per request, and a
    missing/blank secret rejects every request. The gate is **never**
    opened by ``DEV_MODE`` — the slides route is reachable only through
    the ``X-Service-Token`` header, which is compared with
    ``hmac.compare_digest`` (timing-safe).
    """
    expected = os.getenv("SANDBOX_SERVICE_TOKEN", "")
    presented = request.headers.get(SERVICE_TOKEN_HEADER, "")
    if not expected or not presented:
        raise HTTPException(status_code=401, detail="Service token required")
    if not hmac.compare_digest(expected.encode("utf-8"), presented.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Service token required")


def get_current_user(request: Request, db=Depends(get_db)) -> User:
    """Resolve the current user from the sign-in token.

    Always requires a valid token + existing user (not gated by DEV_MODE).
    Used by /auth/me endpoints where the user's own profile is the resource.
    """
    from db import User

    payload = _decode_token(request)
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user_optional(request: Request, db=Depends(get_db)) -> User | None:
    """Resolve the current user, or None when absent/unsigned (never raises).

    Used by the probe path to attribute server-side review cards: in DEV_MODE
    with no real user present this returns None (probe still works, no card),
    and with a signed-in user it returns that user.
    """
    from db import User

    try:
        payload = _decode_token(request)
    except HTTPException:
        return None
    return db.query(User).filter(User.email == payload["sub"]).first()
