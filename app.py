"""
app.py
------
PocketSmart AI: Your Smart Budget & Recommendation Assistant.
FastAPI backend tying together auth, the three planners, session
tracking, and recommendation history.

Run with:  python app.py   (or)  uvicorn app:app --reload
"""
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from auth_utils import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    RegisterUser,
    UserInDB,
    UserSession,
    active_sessions,
    authenticate_user,
    blacklisted_tokens,
    create_access_token,
    create_user,
    get_current_active_user,
    get_current_user,
    get_token,
)
from gemini_utils import (
    HomeBudgetInput,
    JewelryBudgetInput,
    PartyBudgetInput,
    get_home_recommendations,
    get_jewelry_recommendations,
    get_party_recommendations,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="PocketSmart AI: Your Smart Budget Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/uploads", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory recommendation history: username -> list of records
class HistoryRecord(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: dict
    result_summary: dict
    full_result: dict


user_recommendations: Dict[str, List[HistoryRecord]] = {}


def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict) -> None:
    record = HistoryRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        recommendation_type=recommendation_type,
        input_summary=input_data,
        result_summary={
            "total_budget": result.get("total_budget"),
            "remaining_budget": result.get("remaining_budget"),
        },
        full_result=result,
    )
    user_recommendations.setdefault(username, []).append(record)


def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename or "")[1] or ".jpg"
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join("static", "uploads", filename)
    with open(path, "wb") as f:
        f.write(upload_file.file.read())
    return path


# ---------------------------------------------------------------------------
# Pages: home, auth
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    token = await get_token(request)
    user = await get_current_user(request, token)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    token = await get_token(request)
    user = await get_current_user(request, token)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    full_name: Optional[str] = Form(None),
):
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    create_user(RegisterUser(username=username, email=email, password=password, full_name=full_name))
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2-style JSON login endpoint (used by the login form via fetch())."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    existing_user_data = {}
    if user.username in active_sessions:
        existing_user_data = active_sessions[user.username].user_data
        blacklisted_tokens.add(active_sessions[user.username].token)

    active_sessions[user.username] = UserSession(
        username=user.username,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token,
        user_data=existing_user_data,
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            from jose import jwt
            from auth_utils import SECRET_KEY, ALGORITHM

            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except Exception:
            pass

    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    recent = sorted(
        user_recommendations.get(current_user.username, []),
        key=lambda r: r.timestamp,
        reverse=True,
    )[:5]
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "user": current_user, "recent": recent}
    )


# ---------------------------------------------------------------------------
# Scenario 1: Home Interior Planner
# ---------------------------------------------------------------------------
@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse("home_planner.html", {"request": request, "user": current_user})


@app.post("/generate-home", response_class=HTMLResponse)
async def generate_home(
    request: Request,
    total_budget: float = Form(...),
    num_lights: int = Form(0),
    num_fans: int = Form(0),
    num_furniture: int = Form(0),
    num_dining_tables: int = Form(0),
    has_living_room: bool = Form(False),
    has_kitchen: bool = Form(False),
    has_bedroom: bool = Form(False),
    additional_requirements: str = Form(""),
    current_user: UserInDB = Depends(get_current_active_user),
):
    budget_input = HomeBudgetInput(
        total_budget=total_budget,
        num_lights=num_lights,
        num_fans=num_fans,
        num_furniture=num_furniture,
        num_dining_tables=num_dining_tables,
        has_living_room=has_living_room,
        has_kitchen=has_kitchen,
        has_bedroom=has_bedroom,
        additional_requirements=additional_requirements or None,
    )

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
        }

    result = get_home_recommendations(budget_input)
    save_to_history(current_user.username, "home", budget_input.dict(), result)

    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "user": current_user, "result": result, "plan_type": "Home Interior"},
    )


# ---------------------------------------------------------------------------
# Scenario 2: Party Planner
# ---------------------------------------------------------------------------
@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse("party_planner.html", {"request": request, "user": current_user})


@app.post("/generate-party", response_class=HTMLResponse)
async def generate_party(
    request: Request,
    total_budget: float = Form(...),
    num_guests: int = Form(...),
    party_type: str = Form(...),
    venue_type: str = Form(""),
    needs_catering: bool = Form(False),
    needs_decoration: bool = Form(False),
    needs_entertainment: bool = Form(False),
    additional_requirements: str = Form(""),
    current_user: UserInDB = Depends(get_current_active_user),
):
    budget_input = PartyBudgetInput(
        total_budget=total_budget,
        num_guests=num_guests,
        party_type=party_type,
        venue_type=venue_type or None,
        needs_catering=needs_catering,
        needs_decoration=needs_decoration,
        needs_entertainment=needs_entertainment,
        additional_requirements=additional_requirements or None,
    )

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
            "guests": budget_input.num_guests,
        }

    result = get_party_recommendations(budget_input)
    save_to_history(current_user.username, "party", budget_input.dict(), result)

    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "user": current_user, "result": result, "plan_type": "Party"},
    )


# ---------------------------------------------------------------------------
# Scenario 3: Jewelry Planner (with optional outfit image upload)
# ---------------------------------------------------------------------------
@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse("jewelry_planner.html", {"request": request, "user": current_user})


@app.post("/generate-jewelry", response_class=HTMLResponse)
async def generate_jewelry(
    request: Request,
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: str = Form(""),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_active_user),
):
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences or None,
    )

    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image is not None,
        }

    result = get_jewelry_recommendations(budget_input, image_path)

    input_data = budget_input.dict()
    if image_path:
        input_data["image"] = os.path.basename(image_path)
    save_to_history(current_user.username, "jewelry", input_data, result)

    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "user": current_user, "result": result, "plan_type": "Jewelry"},
    )


# ---------------------------------------------------------------------------
# History & recommendation details
# ---------------------------------------------------------------------------
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    records = sorted(
        user_recommendations.get(current_user.username, []),
        key=lambda r: r.timestamp,
        reverse=True,
    )
    return templates.TemplateResponse(
        "history.html", {"request": request, "user": current_user, "history": records}
    )


@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str, current_user: UserInDB = Depends(get_current_active_user)
):
    records = user_recommendations.get(current_user.username, [])
    for item in records:
        if item.id == recommendation_id:
            return item
    raise HTTPException(status_code=404, detail="Recommendation not found")


# ---------------------------------------------------------------------------
# Session info endpoints
# ---------------------------------------------------------------------------
@app.get("/session-info")
async def get_session_info(current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        session = active_sessions[current_user.username]
        return {
            "username": session.username,
            "login_time": session.login_time,
            "last_activity": session.last_activity,
            "session_duration_minutes": (datetime.utcnow() - session.login_time).total_seconds() // 60,
            "user_data": session.user_data,
        }
    raise HTTPException(status_code=404, detail="No active session found")


@app.post("/session-data")
async def update_session_data(data: dict, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data.update(data)
        active_sessions[current_user.username].last_activity = datetime.utcnow()
        return {"message": "Session data updated", "data": active_sessions[current_user.username].user_data}
    raise HTTPException(status_code=404, detail="No active session found")


# ---------------------------------------------------------------------------
# Startup: background cleanup of expired sessions
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def setup_session_cleanup():
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired = [
                username
                for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 min
            ]
            for username in expired:
                del active_sessions[username]
            await asyncio.sleep(300)  # check every 5 minutes

    asyncio.create_task(cleanup_expired_sessions())


if __name__ == "__main__":
    import uvicorn

    print("Starting PocketSmart AI: Smart Budget Planner...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
