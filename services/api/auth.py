"""Password hashing + JWT helpers."""
import os
import time
from dataclasses import dataclass

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = int(os.environ.get("TOKEN_TTL_SECONDS", str(8 * 3600)))
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _secret() -> str:
    s = os.environ.get("JWT_SECRET", "")
    if len(s) < 32:
        raise RuntimeError("JWT_SECRET must be set and at least 32 characters")
    return s


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


DUMMY_HASH = hash_password("dummy-password-for-timing")  # equalizes timing for unknown users


def create_token(username: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + TOKEN_TTL_SECONDS},
                      _secret(), algorithm=ALGORITHM)


@dataclass
class User:
    username: str
    role: str


def current_user(token: str = Depends(oauth2)) -> User:
    try:
        claims = jwt.decode(token, _secret(), algorithms=[ALGORITHM])  # algorithm pinned
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return User(claims["sub"], claims["role"])


def require_role(*roles: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "Insufficient role")
        return user
    return dep
