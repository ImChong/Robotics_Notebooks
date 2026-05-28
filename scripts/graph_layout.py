"""Offline force-directed layout for wiki link-graph (aligned with docs/graph.html defaults)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# Match docs/graph.html initial physics (charge slider 800 → strength -800).
LAYOUT_VERSION = 1
LAYOUT_WIDTH = 1200
LAYOUT_HEIGHT = 800
CHARGE_STRENGTH = -800.0
LINK_DISTANCE = 80.0
LINK_STRENGTH = 0.35
CENTER_STRENGTH = 0.05
DISTANCE_MAX = 400.0
COLLISION_STRENGTH = 0.7
ALPHA_DECAY = 0.025
MAX_TICKS = 320
RNG_SEED = 42

LAYOUT_PARAMS: dict[str, float | int] = {
    "version": LAYOUT_VERSION,
    "width": LAYOUT_WIDTH,
    "height": LAYOUT_HEIGHT,
    "charge": CHARGE_STRENGTH,
    "link_distance": LINK_DISTANCE,
    "link_strength": LINK_STRENGTH,
    "center_strength": CENTER_STRENGTH,
    "distance_max": DISTANCE_MAX,
    "collision_strength": COLLISION_STRENGTH,
    "alpha_decay": ALPHA_DECAY,
    "node_scale": 1.0,
}

_D3_LAYOUT_SCRIPT = Path(__file__).with_name("compute_force_layout.mjs")


def base_radius(degree: float, node_scale: float = 1.0) -> float:
    return max(5.0, min(18.0, 4.0 + float(max(degree, 0.0)) ** 0.5 * 2.2)) * node_scale


def compute_force_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
    degree_map: dict[str, int],
    *,
    width: int = LAYOUT_WIDTH,
    height: int = LAYOUT_HEIGHT,
) -> dict[str, Any]:
    """Run d3-force (Barnes-Hut) and return layout metadata for link-graph.json."""
    if not nodes:
        return {
            "version": LAYOUT_VERSION,
            "width": width,
            "height": height,
            "params": dict(LAYOUT_PARAMS),
            "positions": {},
        }

    payload = {
        "nodes": [{"id": node["id"]} for node in nodes],
        "edges": edges,
        "degree_map": degree_map,
        "width": width,
        "height": height,
        "seed": RNG_SEED,
        "params": dict(LAYOUT_PARAMS),
        "max_ticks": MAX_TICKS,
    }
    proc = subprocess.run(
        ["node", str(_D3_LAYOUT_SCRIPT)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        cwd=_D3_LAYOUT_SCRIPT.parent.parent,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "d3-force layout failed (is Node installed and `npm ci` run?). "
            f"exit={proc.returncode}: {detail[:500]}"
        ) from None
    layout = json.loads(proc.stdout)
    layout["width"] = width
    layout["height"] = height
    return layout
