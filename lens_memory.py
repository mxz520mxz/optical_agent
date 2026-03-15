"""
Lens configuration memory module.

Stores full lens_config dicts on disk and exposes short IDs for use in
LLM message history. This keeps the context window small — messages only
carry a short ID like "lens_a1b2c3d4" instead of 2000+ chars of JSON.

Usage:
    import lens_memory

    lens_id = lens_memory.store(lens_config)          # save, get short ID
    cfg     = lens_memory.load(lens_id)               # retrieve by ID
    summary = lens_memory.summarize(lens_config)      # compact text description
"""

import json
import math
import os
import uuid

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "memory")


# ── Public API ─────────────────────────────────────────────────────────────────

def store(lens_config: dict, label: str | None = None) -> str:
    """
    Save lens_config to disk and return a lens_id.

    Args:
        lens_config: Full lens configuration dict.
        label: Optional explicit ID. If not given, a random 8-char hex ID is used.

    Returns:
        lens_id string (e.g. "lens_a1b2c3d4" or the given label).
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    lens_id = label if label else f"lens_{uuid.uuid4().hex[:8]}"
    path = os.path.join(MEMORY_DIR, f"{lens_id}.json")
    with open(path, "w") as f:
        json.dump(lens_config, f, indent=2)
    return lens_id


def _clean_deeplens_json(data):
    """
    递归清洗 DeepLens 导出的 JSON。
    去掉所有键名两侧的括号，例如将 "(c)" 还原为 "c"。
    """
    if isinstance(data, dict):
        clean_dict = {}
        for k, v in data.items():
            # 剥离键名两端的 '(' 和 ')'
            clean_k = k.strip("()")
            clean_dict[clean_k] = _clean_deeplens_json(v)
        return clean_dict
    elif isinstance(data, list):
        return [_clean_deeplens_json(item) for item in data]
    else:
        return data
    
def load(lens_id: str) -> dict:
    """
    Load a lens_config by its ID.

    Raises:
        ValueError: If the lens_id is not found in memory.
    """
    path = os.path.join(MEMORY_DIR, f"{lens_id}.json")
    if not os.path.exists(path):
        raise ValueError(
            f"Lens ID '{lens_id}' not found in memory. "
            f"Available IDs: {list_ids()}"
        )
    with open(path) as f:
        return _clean_deeplens_json(json.load(f))


def list_ids() -> list[str]:
    """Return all stored lens IDs."""
    if not os.path.isdir(MEMORY_DIR):
        return []
    return [f[:-5] for f in os.listdir(MEMORY_DIR) if f.endswith(".json")]


def summarize(lens_config: dict) -> str:
    """
    Generate a compact one-line text summary of a lens_config.

    Example output:
        "Triplet | 50.0mm f/2.8 | FOV 46°| 5 surfaces | 3 elements"
    """
    surfaces = lens_config.get("surfaces", [])
    num_surfaces = len(surfaces)

    # Count refractive elements (pairs of air-glass-air transitions)
    num_elements = _count_elements(surfaces)

    foclen = lens_config.get("foclen", lens_config.get("focal_length", 0.0))
    fnum   = lens_config.get("fnum", 0.0)
    fov    = lens_config.get("fov", lens_config.get("hfov", 0.0))
    if fov and fov < 90:  # half-FOV stored
        fov_label = f"FOV {fov*2:.0f}°"
    elif fov:
        fov_label = f"FOV {fov:.0f}°"
    else:
        fov_label = ""

    fnum_str = f"f/{fnum:.1f}" if fnum else ""
    fl_str   = f"{foclen:.1f}mm" if foclen else ""

    parts = [
        f"{fl_str} {fnum_str}".strip(),
        fov_label,
        f"{num_surfaces} surfaces",
        f"{num_elements} elements",
    ]
    return " | ".join(p for p in parts if p)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_elements(surfaces: list) -> int:
    """
    Estimate number of glass elements from surface list.
    Each element has 2 refractive surfaces (front + rear).
    """
    glass_surfaces = sum(
        1 for s in surfaces
        if s.get("mat2", "air").lower() not in ("air", "", "vacuum")
        or s.get("mat1", "air").lower() not in ("air", "", "vacuum")
    )
    # Rough estimate: each element contributes ~2 surfaces
    return max(1, math.ceil(glass_surfaces / 2)) if glass_surfaces else max(1, len(surfaces) // 2)
