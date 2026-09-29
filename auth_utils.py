"""
Authentication utilities for PocketSmart AI.

Uses a simple JSON file as a user "database" (fine for a demo/project;
swap for a real DB like SQLite/Postgres in production).
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

USERS_FILE = os.path.join(os.path.dirname(__file__), "users_db.json")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory session store: username -> {login_time, last_activity, user_data}
active_sessions = {}
blacklisted_tokens = set()


class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str


class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: dict = {}


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_user(username: str) -> Optional[UserInDB]:
    users = _load_users()
    if username in users:
        return UserInDB(**users[username])
    return None


def create_user(user: RegisterUser) -> UserInDB:
    users = _load_users()
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed = pwd_context.hash(user.password)
    record = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed,
    }
    users[user.username] = record
    _save_users(users)
    return UserInDB(**record)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user:
        return None
    if not pwd_context.verify(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_token(request: Request) -> Optional[str]:
    """Read the JWT from the httpOnly cookie set at login."""
    return request.cookies.get("access_token")


async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> Optional[UserInDB]:
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return get_user(username)


async def get_current_active_user(request: Request) -> UserInDB:
    """Dependency that protects a route: 401s if not logged in."""
    token = await get_token(request)
    user = await get_current_user(request, token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"Location": "/login"},
        )
    return user
