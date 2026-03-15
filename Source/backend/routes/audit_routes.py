from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

try:
    from ..database_api import query_audit_logs
    from ..rbac_api import require_roles
    from ..wm_common import _log_audit
except ImportError:
    from database_api import query_audit_logs  # type: ignore
    from rbac_api import require_roles  # type: ignore
    from wm_common import _log_audit  # type: ignore


def register_audit_routes(app: Flask) -> None:
    @app.route("/audit/logs", methods=["GET"])
    @require_roles("AUDITOR", "ADMIN")
    def audit_logs_list() -> tuple[Any, int]:
        try:
            limit = min(int(request.args.get("limit", 100)), 500)
            offset = max(int(request.args.get("offset", 0)), 0)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "limit and offset must be integers"}), 400

        actor_filter = (request.args.get("actor_user_id") or "").strip() or None
        action_filter = (request.args.get("action") or "").strip() or None
        outcome_filter = (request.args.get("outcome") or "").strip() or None
        resource_filter = (request.args.get("resource_id") or "").strip() or None
        from_ts = (request.args.get("from") or "").strip() or None
        to_ts = (request.args.get("to") or "").strip() or None

        rows, total = query_audit_logs(
            limit=limit,
            offset=offset,
            actor_user_id=actor_filter,
            action=action_filter,
            outcome=outcome_filter,
            resource_id=resource_filter,
            from_ts=from_ts,
            to_ts=to_ts,
        )

        _log_audit(
            "AUDIT_LOGS_ACCESSED", "SUCCESS",
            detail={"limit": limit, "offset": offset, "filters": {
                "actor_user_id": actor_filter,
                "action": action_filter,
                "outcome": outcome_filter,
                "resource_id": resource_filter,
            }},
        )

        return jsonify({
            "status": "success",
            "logs": rows,
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200
