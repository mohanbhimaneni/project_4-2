from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify

try:
    from .database_api import close_db, db_api, init_db
    from .rbac_api import rbac_api
    from .routes.audit_routes import register_audit_routes
    from .routes.study_routes import register_study_routes
    from .routes.watermark_routes import register_watermark_routes
    from .wm_common import CURRENT_DIR
except ImportError:
    from database_api import close_db, db_api, init_db  # type: ignore
    from rbac_api import rbac_api  # type: ignore
    from routes.audit_routes import register_audit_routes  # type: ignore
    from routes.study_routes import register_study_routes  # type: ignore
    from routes.watermark_routes import register_watermark_routes  # type: ignore
    from wm_common import CURRENT_DIR  # type: ignore


def create_app() -> Flask:
    app = Flask(__name__)

    storage_root = CURRENT_DIR / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    app.config["SECRET_KEY"] = os.environ.get("SECUREDICOM_SECRET", "securedicom-dev-secret")
    app.config["TOKEN_MAX_AGE_SECONDS"] = int(os.environ.get("SECUREDICOM_TOKEN_TTL", "86400"))
    app.config["DATABASE_PATH"] = str(storage_root / "securedicom.db")
    app.config["STORAGE_ROOT"] = str(storage_root)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    init_db(Path(app.config["DATABASE_PATH"]))

    app.register_blueprint(rbac_api)
    app.register_blueprint(db_api)

    @app.teardown_appcontext
    def _teardown(_: Any) -> None:
        close_db()

    @app.after_request
    def _cors(resp: Any) -> Any:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return resp

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok", "service": "securedicom-backend"}), 200

    register_watermark_routes(app)
    register_study_routes(app)
    register_audit_routes(app)

    return app
