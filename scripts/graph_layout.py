"""Offline force-directed layout for wiki link-graph (aligned with docs/graph.html defaults)."""

from __future__ import annotations

from typing import Any

import numpy as np

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
VELOCITY_DECAY = 0.6
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
    """Run a force simulation and return layout metadata for link-graph.json."""
    n = len(nodes)
    if n == 0:
        return {
            "version": LAYOUT_VERSION,
            "width": width,
            "height": height,
            "params": dict(LAYOUT_PARAMS),
            "positions": {},
        }

    id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}
    radii = np.array(
        [base_radius(float(degree_map.get(node["id"], 0))) + 5.0 for node in nodes],
        dtype=np.float64,
    )

    rng = np.random.default_rng(RNG_SEED)
    pos = rng.uniform(-40.0, 40.0, (n, 2))
    pos[:, 0] += width / 2.0
    pos[:, 1] += height / 2.0
    vel = np.zeros((n, 2), dtype=np.float64)

    edge_i: list[int] = []
    edge_j: list[int] = []
    for edge in edges:
        si = id_to_idx.get(edge["source"])
        ti = id_to_idx.get(edge["target"])
        if si is None or ti is None:
            continue
        edge_i.append(si)
        edge_j.append(ti)
    edge_i_arr = np.asarray(edge_i, dtype=np.intp)
    edge_j_arr = np.asarray(edge_j, dtype=np.intp)

    cx, cy = width / 2.0, height / 2.0
    alpha = 1.0

    for _ in range(MAX_TICKS):
        if alpha < 0.001:
            break

        disp = np.zeros((n, 2), dtype=np.float64)

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(dist > 0, 1.0 / dist, 0.0)
            force_mag = np.where(
                dist < DISTANCE_MAX,
                CHARGE_STRENGTH * alpha * inv,
                0.0,
            )
        force = diff * force_mag[..., np.newaxis]
        disp += np.sum(force, axis=1)

        if edge_i_arr.size:
            delta = pos[edge_j_arr] - pos[edge_i_arr]
            dist_e = np.linalg.norm(delta, axis=1) + 1e-6
            pull = ((dist_e - LINK_DISTANCE) / dist_e) * LINK_STRENGTH * alpha
            fvec = delta * pull[:, np.newaxis]
            np.add.at(disp, edge_i_arr, fvec)
            np.add.at(disp, edge_j_arr, -fvec)

        disp[:, 0] += (cx - pos[:, 0]) * CENTER_STRENGTH * alpha
        disp[:, 1] += (cy - pos[:, 1]) * CENTER_STRENGTH * alpha

        diff_c = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_c = np.linalg.norm(diff_c, axis=2)
        np.fill_diagonal(dist_c, np.inf)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, min_dist - dist_c)
        safe_dist = np.where(dist_c < np.inf, np.maximum(dist_c, 1e-6), np.inf)
        push = np.where(
            dist_c < np.inf,
            (overlap / safe_dist) * COLLISION_STRENGTH * alpha,
            0.0,
        )
        disp += np.sum(diff_c * push[..., np.newaxis], axis=1)

        vel = (vel + disp) * VELOCITY_DECAY
        pos += vel
        alpha *= 1.0 - ALPHA_DECAY

    positions: dict[str, dict[str, float]] = {}
    for idx, node in enumerate(nodes):
        positions[node["id"]] = {
            "x": round(float(pos[idx, 0]), 2),
            "y": round(float(pos[idx, 1]), 2),
        }

    return {
        "version": LAYOUT_VERSION,
        "width": width,
        "height": height,
        "params": dict(LAYOUT_PARAMS),
        "positions": positions,
    }
