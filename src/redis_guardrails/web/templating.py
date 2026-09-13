import math
from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_GAUGE_CENTER_X = 110
_GAUGE_CENTER_Y = 116
_GAUGE_ARC_RADIUS = 96
_GAUGE_NEEDLE_RADIUS = 78


def gauge_geometry(distance: float) -> dict:
    """SVG coordinates for the match-strength gauge: a semicircle track running
    left (0%) -> top -> right (100%), where 100% is an exact-text match (distance 0)
    and 0% is at or past the maximum guardrail threshold (distance 2.0)."""
    pct = round((1 - min(distance, 2.0) / 2.0) * 100)
    theta = math.radians(180 - (pct / 100) * 180)
    return {
        "pct": pct,
        "arc_x": _GAUGE_CENTER_X + _GAUGE_ARC_RADIUS * math.cos(theta),
        "arc_y": _GAUGE_CENTER_Y - _GAUGE_ARC_RADIUS * math.sin(theta),
        "needle_x": _GAUGE_CENTER_X + _GAUGE_NEEDLE_RADIUS * math.cos(theta),
        "needle_y": _GAUGE_CENTER_Y - _GAUGE_NEEDLE_RADIUS * math.sin(theta),
    }


templates.env.globals["gauge_geometry"] = gauge_geometry
