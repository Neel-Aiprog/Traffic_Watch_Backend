"""
main.py — FastAPI backend for PS2 Event-Driven Congestion System.

Run with:  uvicorn main:app --reload --port 8000
Requires:  pip install fastapi uvicorn[standard] lightgbm xgboost catboost pandas numpy bcrypt

Endpoints:
  POST /access/verify   -> username/password verification, returns session token
  POST /register        -> register new user
  POST /logout          -> logout user (invalidate session)
  POST /predict         -> takes a raw event, returns prediction + recommendation
  GET  /health          -> simple liveness check
"""

import os
import time
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from explain import explain_prediction
from inference import predict_severity
from recommendation import get_recommendation
from sqlite_auth import get_auth_manager, init_default_users
load_dotenv()
# Initialize SQLite auth manager
try:
    auth_manager = get_auth_manager()
    print("[INFO] SQLiteAuth manager initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize SQLiteAuth manager: {e}")
    auth_manager = None

# Initialize default users on startup
if auth_manager:
    try:
        init_default_users()
    except Exception as e:
        print(f"[WARN] Error initializing default users: {e}")

# ── Schemas ────────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Operator username")
    password: str = Field(..., min_length=6, max_length=100, description="Operator password")


class LoginResponse(BaseModel):
    token: str = Field(..., description="Session token for authenticated requests")
    username: str = Field(..., description="Authenticated operator username")
    role: str = Field(..., description="Operator role")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=6, max_length=100, description="Password")
    role: str = Field(default="operator", description="User role (operator, supervisor, admin)")


class RegisterResponse(BaseModel):
    success: bool = Field(..., description="Whether registration was successful")
    message: str = Field(..., description="Status message")
    username: Optional[str] = Field(None, description="Username if registration successful")


class EventInput(BaseModel):
    event_id: Optional[str] = Field(None, description="Optional event identifier")
    event_cause: str = Field(..., description="Cause of the event (e.g., 'festival', 'construction')")
    corridor: str = Field(..., description="Affected corridor or road segment")
    priority: str = Field(..., description="Priority level (LOW, MEDIUM, HIGH)")
    requires_road_closure: bool = Field(..., description="Whether road closure is required")
    veh_type: Optional[str] = Field(None, description="Types of vehicles involved")
    start_datetime: str = Field(..., description="Event start time in ISO 8601 format")


# ── Authentication Dependencies ──────────────────────────────────────────────────────
def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Validate session token and return current user information.

    Expects Authorization header in format: "Bearer <token>"
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )

    user = auth_manager.validate_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ── Application Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PS2 Event-Driven Congestion API",
    description="API for forecasting event-related traffic impact and recommending responses",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ────────────────────────────────────────────────────────────────────
@app.post("/access/verify", response_model=LoginResponse)
def login(login_request: LoginRequest):
    """
    Authenticate operator with username and password.

    Returns a session token that must be included in subsequent requests
    as an Authorization header: "Bearer <token>"
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )

    # Validate credentials
    user = auth_manager.validate_user(
        login_request.username,
        login_request.password
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create session
    token, expires_at = auth_manager.create_session(user["id"])

    return LoginResponse(
        token=token,
        username=user["username"],
        role=user["role"]
    )


@app.post("/register", response_model=RegisterResponse)
def register(register_request: RegisterRequest):
    """
    Register a new user.

    Returns success status and message.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )

    # Check if username already exists
    existing_user = auth_manager.get_user_by_username(register_request.username)
    if existing_user:
        return RegisterResponse(
            success=False,
            message="Username already exists",
            username=None
        )

    # Create new user
    success = auth_manager.create_user(
        username=register_request.username,
        password=register_request.password,
        role=register_request.role
    )

    if success:
        return RegisterResponse(
            success=True,
            message="User registered successfully",
            username=register_request.username
        )
    else:
        return RegisterResponse(
            success=False,
            message="Registration failed due to unknown error",
            username=None
        )


@app.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """
    Logout user by invalidating session.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization header"
        )

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization header format. Use 'Bearer <token>'"
        )

    # Invalidate session
    success = auth_manager.invalidate_session(token)

    if success:
        return {"message": "Logged out successfully"}
    else:
        return {"message": "No active session to logout"}


@app.get("/health")
def health():
    """Health check endpoint."""
    db_status = "connected" if auth_manager else "disconnected"
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status
    }


@app.post("/predict")
def predict(event: EventInput, current_user: dict = Depends(get_current_user)):
    """
    Main prediction endpoint - requires authentication.

    Takes an event description and returns:
    - Prediction of severity tier
    - Recommended response actions
    - Debug information (in development)
    """
    event_dict = event.model_dump()

    try:
        prediction = predict_severity(event_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    # Extract and strip the internal feature row before building the response
    feature_row = prediction.pop("_feature_row")

    recommendation = get_recommendation(
        prediction["tier"],
        event.corridor
    )

    # Compute per-prediction SHAP explanation — runs fast on a single row
    try:
        explanation_factors = explain_prediction(event_dict, feature_row)
    except Exception as e:
        # Explanation failure must never take down the prediction response
        print(f"[predict] Explanation failed (non-fatal): {e}")
        explanation_factors = []

    return {
        "event_id": event.event_id or f"evt_{int(time.time())}",
        "received_at": event.start_datetime,
        "operator": current_user["username"],
        "input_summary": {
            "event_cause": event.event_cause,
            "corridor": event.corridor,
            "priority": event.priority,
            "requires_road_closure": event.requires_road_closure,
            "veh_type": event.veh_type,
        },
        "prediction": {
            "tier": prediction["tier"],
            "probability_high": prediction["probability_high"],
            "thresholds_used": prediction["thresholds_used"],
        },
        "recommendation": recommendation,
        "model_debug": {
            "individual_scores": prediction["individual_scores"],
        },
        "explanation": {
            "factors": explanation_factors,
        },
    }


# ── Startup Event ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Perform startup tasks."""
    if auth_manager:
        # Cleanup expired sessions on startup
        auth_manager.cleanup_expired_sessions()
        print("[INFO] Cleaned up expired sessions on startup")