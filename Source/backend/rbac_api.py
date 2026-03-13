from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from flask import Blueprint, current_app, g, jsonify, request
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from .database_api import create_user, get_user_by_email, get_user_by_id
except ImportError:
    from database_api import create_user, get_user_by_email, get_user_by_id  # type: ignore


rbac_api = Blueprint("rbac_api", __name__, url_prefix="/auth")

ALLOWED_ROLES = {"DOCTOR", "PATIENT", "ADMIN"}


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def generate_token(user_id: str, role: str) -> str:
    return _serializer().dumps({"user_id": user_id, "role": role})


def decode_token(token: str, max_age_seconds: int) -> Optional[dict[str, Any]]:
    try:
        payload = _serializer().loads(token, max_age=max_age_seconds)
        return payload
    except (BadSignature, SignatureExpired):
        return None


def _extract_bearer_token() -> Optional[str]:
    query_token = request.args.get("token", "").strip()
    if query_token:
        return query_token

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header.replace("Bearer ", "", 1).strip()


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _extract_bearer_token()
        if not token:
            return jsonify({"status": "error", "message": "Missing bearer token"}), 401

        payload = decode_token(token, max_age_seconds=current_app.config["TOKEN_MAX_AGE_SECONDS"])
        if not payload:
            return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

        user = get_user_by_id(payload["user_id"])
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 401

        g.current_user = user
        return func(*args, **kwargs)

    return wrapper


def require_roles(*roles: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    allowed = {r.upper() for r in roles}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        @require_auth
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user = g.current_user
            if user["role"].upper() not in allowed:
                return jsonify({"status": "error", "message": "Forbidden for this role"}), 403
            return func(*args, **kwargs)

        return wrapper

    return decorator


@rbac_api.route("/signup", methods=["POST"])
def signup() -> tuple[Any, int]:
    body = request.get_json(silent=True) or {}

    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    role = (body.get("role") or "PATIENT").strip().upper()

    if not name or len(name) < 2:
        return jsonify({"status": "error", "message": "Name must be at least 2 characters"}), 400
    if "@" not in email:
        return jsonify({"status": "error", "message": "Invalid email"}), 400
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must be at least 8 characters"}), 400
    if role not in ALLOWED_ROLES:
        return jsonify({"status": "error", "message": f"Invalid role. Allowed: {sorted(ALLOWED_ROLES)}"}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"status": "error", "message": "Email already registered"}), 409

    password_hash = generate_password_hash(password)
    user = create_user(name=name, email=email, password_hash=password_hash, role=role)
    token = generate_token(user_id=user["id"], role=user["role"])

    return jsonify({"status": "success", "token": token, "user": user}), 201


@rbac_api.route("/login", methods=["POST"])
def login() -> tuple[Any, int]:
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if "@" not in email or not password:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 400

    user = get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    token = generate_token(user_id=user["id"], role=user["role"])

    return jsonify(
        {
            "status": "success",
            "token": token,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
                "created_at": user["created_at"],
            },
        }
    ), 200


@rbac_api.route("/me", methods=["GET"])
@require_auth
def me() -> tuple[Any, int]:
    user = g.current_user
    return jsonify({"status": "success", "user": user}), 200
