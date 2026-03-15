from __future__ import annotations

import logging

try:
    from .app_factory import create_app
    from .wm_common import (
        _extract_roi_mask_and_overlay,
        _load_model_if_needed,
        _select_working_image,
    )
except ImportError:
    from app_factory import create_app  # type: ignore
    from wm_common import _extract_roi_mask_and_overlay, _load_model_if_needed, _select_working_image  # type: ignore


app = create_app()


__all__ = [
    "app",
    "create_app",
    "_extract_roi_mask_and_overlay",
    "_load_model_if_needed",
    "_select_working_image",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5001, debug=True)
