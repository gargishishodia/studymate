"""
auth.py  —  StudyMate authentication (signup, login, JWT, route protection)

This is the file to understand line-by-line for your interview. It covers:
  - hashing passwords with bcrypt (never store plain text)
  - issuing a signed JWT on login
  - verifying that JWT to protect routes (get_current_user)

The big interview ideas, all visible here:
  AUTHENTICATION (authn) = proving WHO you are -> login + password check
  AUTHORIZATION (authz)  = what you're ALLOWED to do -> the get_current_user
                           dependency gating protected routes

pip install "python-jose[cryptography]" "passlib[bcrypt]"
"""

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import bcrypt
from pydantic import BaseModel
from dotenv import load_dotenv

from database import get_connection

load_dotenv()

# --- Config ---
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"                 # how the token is signed
ACCESS_TOKEN_EXPIRE_MINUTES = 30    # short-lived: limits damage if stolen

# bcrypt is a deliberately SLOW hash, which makes brute-forcing stolen
# password hashes impractical. It also salts automatically.

# This tells FastAPI tokens arrive as "Authorization: Bearer <token>",
# and that /auth/login is where you go to get one. It also makes the
# Authorize button appear in /docs.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# A router groups these endpoints; main.py will include it.
router = APIRouter(prefix="/auth", tags=["auth"])


# --- Request/response shapes ---
class SignupIn(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


# ----------------------------------------------------------------------
# Password helpers
# ----------------------------------------------------------------------
def hash_password(plain: str) -> str:
    """Turn a plain password into a bcrypt hash to store safely.
    bcrypt only handles up to 72 bytes, so we truncate longer passwords."""
    pw = plain.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a login attempt against the stored hash. bcrypt handles the salt."""
    pw = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw, hashed.encode("utf-8"))


# ----------------------------------------------------------------------
# JWT helper
# ----------------------------------------------------------------------
def create_access_token(email: str) -> str:
    """
    Build a signed JWT. A JWT is three base64 parts: header.payload.signature.
    We put the user's email in `sub` (subject) and an expiry in `exp`. The
    signature is an HMAC of header+payload using SECRET_KEY - only someone
    with the secret can produce a valid one, which is the whole security model.
    Note: the payload is readable by anyone (it's not encrypted), so we never
    put the password or anything sensitive in it.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ----------------------------------------------------------------------
# Small DB helpers for users
# ----------------------------------------------------------------------
def get_user_by_email(email: str):
    """Return (id, email, password_hash) for a user, or None if not found."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


# ----------------------------------------------------------------------
# ENDPOINT: signup
# ----------------------------------------------------------------------
@router.post("/signup")
def signup(data: SignupIn):
    # Don't allow duplicate accounts.
    if get_user_by_email(data.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(data.password)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
        (data.email, hashed),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok", "message": "Account created. You can now log in."}


# ----------------------------------------------------------------------
# ENDPOINT: login
# ----------------------------------------------------------------------
@router.post("/login")
def login(data: LoginIn):
    user = get_user_by_email(data.email)
    # Same error whether the email is unknown or the password is wrong, so an
    # attacker can't tell which emails exist (avoids user enumeration).
    if not user or not verify_password(data.password, user[2]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(email=user[1])
    return {"access_token": token, "token_type": "bearer"}


# ----------------------------------------------------------------------
# DEPENDENCY: protect routes
# Any endpoint that adds `user = Depends(get_current_user)` will require a
# valid token. This is the AUTHORIZATION gate.
# ----------------------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Decode and verify the JWT. If the signature is bad or it's expired,
    jwt.decode raises and we return 401. Otherwise we pull the email out of
    the `sub` claim and confirm the user still exists. Returns the email.
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = get_user_by_email(email)
    if user is None:
        raise credentials_error
    return email
