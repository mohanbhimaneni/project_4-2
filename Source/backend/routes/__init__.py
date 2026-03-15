from .audit_routes import register_audit_routes
from .study_routes import register_study_routes
from .watermark_routes import register_watermark_routes

__all__ = [
    "register_audit_routes",
    "register_study_routes",
    "register_watermark_routes",
]
