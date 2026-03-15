"""
Lens design tool implementations.
Wraps DeepLens operations and provides JSON-serializable interfaces
for the Claude agent to call as tools.

Memory model: full lens configurations are stored on disk via lens_memory.
Tools accept and return short lens IDs (e.g. "lens_a1b2c3d4") so that
the LLM message history never contains large JSON blobs.
"""

import copy
import json
import math
import os
import sys
import tempfile
import time
from typing import Any

from lens_templates import select_template
import lens_memory

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── DeepLens availability check ──────────────────────────────────────────────

# Prefer git-cloned DeepLens in ./DeepLens/ over any pip-installed version
_deeplens_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepLens")
if os.path.isdir(_deeplens_local) and _deeplens_local not in sys.path:
    sys.path.insert(0, _deeplens_local)

try:
    import torch
    from deeplens import GeoLens
    DEEPLENS_AVAILABLE = True
except ImportError:
    DEEPLENS_AVAILABLE = False


# ─── Public tool functions (called by agent) ──────────────────────────────────

def design_initial_structure(
    focal_length: float,
    fnum: float,
    fov_deg: float = 40.0,
    sensor_size: list | None = None,
    num_elements: int | None = None,
    lens_type: str | None = None,
    wavelengths: str = "visible",
) -> dict:
    """
    Generate an initial lens structure from optical specifications.

    Returns a dict with:
      - lens_id: ID to reference this lens in subsequent tool calls
      - template_used: name of starting template
      - num_elements: number of glass elements
      - summary: human-readable description
    """
    sensor_tuple = tuple(sensor_size) if sensor_size else None

    override_map = {
        "singlet": 1, "doublet": 2, "triplet": 3,
        "double_gauss": 6, "telephoto": 5, "wide_angle": 6,
    }
    if lens_type and num_elements is None:
        num_elements = override_map.get(lens_type.lower().replace(" ", "_"))

    lens_config = select_template(
        focal_length=focal_length,
        fnum=fnum,
        fov_deg=fov_deg,
        sensor_size=sensor_tuple,
        num_elements=num_elements,
    )

    glass_surfaces = [s for s in lens_config["surfaces"] if s["type"] != "Aperture"]
    n_elements = sum(
        1 for i, s in enumerate(glass_surfaces)
        if s["mat2"] != "air" and (i == 0 or glass_surfaces[i - 1]["mat2"] == "air")
    )

    lens_id = lens_memory.store(lens_config)
    summary = (
        f"Designed {n_elements}-element lens: "
        f"f={focal_length:.1f}mm, f/{fnum}, FoV={fov_deg:.0f}°. "
        f"Sensor: {lens_config['sensor_size'][0]:.1f}×"
        f"{lens_config['sensor_size'][1]:.1f}mm"
    )
    return {
        "lens_id": lens_id,
        "template_used": lens_config.get("info", ""),
        "num_elements": n_elements,
        "num_surfaces": len(lens_config["surfaces"]),
        "config_summary": lens_memory.summarize(lens_config),
        "summary": summary,
    }


def run_optimization(
    lens_id: str,
    iterations: int = 2000,
    learning_rates: list | None = None,
) -> dict:
    """
    Run RMS-based gradient optimization on a stored lens (DeepLens).

    Uses a manual PyTorch training loop (loss_rms + CosineAnnealing),
    following the pattern from DeepLens/2_autolens_rms.py.

    Args:
        lens_id: ID of the lens to optimize
        iterations: Number of gradient descent iterations
        learning_rates: [c, d, k, a] learning rates for Adam.
                        Default: [1e-4, 1e-4, 1e-2, 1e-4]

    Returns dict with:
      - lens_id: ID of the optimized lens (new ID)
      - summary: result description
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    lrs = learning_rates or [1e-4, 1e-4, 1e-2, 1e-4]

    if not DEEPLENS_AVAILABLE:
        return _mock_optimization(lens_id, lens_config, iterations)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
        json.dump(lens_config, tmp)
        tmp_path = tmp.name

    result_dir = os.path.join(RESULTS_DIR, f"opt_{int(time.time())}")
    os.makedirs(result_dir, exist_ok=True)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lens = GeoLens(filename=tmp_path, device=device)

        # ── Optimizer & scheduler (same pattern as 2_autolens_rms.py) ──────
        optimizer = lens.get_optimizer(lrs, optim_mat=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, iterations // 4), T_mult=1
        )

        # ── Training loop: minimise RMS spot error only ─────────────────────
        for _ in range(iterations):
            # loss_rms() returns avg RMS tensor of shape (num_grid, num_grid)
            l_rms = lens.loss_rms()
            L = l_rms.mean()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            scheduler.step()

        # ── Save result ──────────────────────────────────────────────────────
        out_json = os.path.join(result_dir, "optimized.json")
        lens.write_lens_json(out_json)
        with open(out_json) as f:
            optimized_config = json.load(f)

        new_id = lens_memory.store(optimized_config)
        return {
            "lens_id": new_id,
            "result_dir": result_dir,
            "iterations_run": iterations,
            "status": "success",
            "config_summary": lens_memory.summarize(optimized_config),
            "summary": f"Optimization complete ({iterations} iterations). Use lens_id={new_id!r} for next steps.",
        }
    except Exception as e:
        return {
            "lens_id": lens_id,
            "status": "error",
            "error": str(e),
            "summary": f"Optimization failed: {e}",
        }
    finally:
        os.unlink(tmp_path)


def evaluate_lens(lens_id: str) -> dict:
    """
    Evaluate lens optical performance using DeepLens analysis.

    Args:
        lens_id: ID of the lens to evaluate

    Returns metrics dict with RMS spot sizes, estimated MTF, distortion, etc.
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    if not DEEPLENS_AVAILABLE:
        return _mock_evaluation(lens_config)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
        json.dump(lens_config, tmp)
        tmp_path = tmp.name

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lens = GeoLens(filename=tmp_path, device=device)
        metrics = _extract_metrics(lens)
        metrics["lens_id"] = lens_id
        metrics["status"] = "success"
        return metrics
    except Exception as e:
        return {
            "lens_id": lens_id,
            "status": "error",
            "error": str(e),
            "summary": f"Evaluation failed: {e}",
        }
    finally:
        os.unlink(tmp_path)


def add_lens_element(
    lens_id: str,
    position: str = "rear",
    material: str = "1.5168/64.2",
    element_type: str = "meniscus",
) -> dict:
    """
    Insert a new lens element into a stored lens structure.

    Args:
        lens_id: ID of the current lens
        position: 'front', 'rear', or 'middle' (before aperture stop)
        material: Glass material in 'n/V' format
        element_type: 'biconvex', 'biconcave', 'meniscus', 'plano_convex'

    Returns dict with new lens_id after insertion.
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    config = copy.deepcopy(lens_config)
    surfaces = config["surfaces"]
    foclen = config.get("foclen", 50.0)
    fnum = config.get("fnum", 2.8)

    scale = foclen / 50.0
    weak_curvature = 0.01 / scale
    thickness = 3.0 * scale

    curvature_sets = {
        "biconvex":    (weak_curvature, -weak_curvature),
        "biconcave":   (-weak_curvature, weak_curvature),
        "meniscus":    (weak_curvature, weak_curvature * 0.8),
        "plano_convex": (weak_curvature, 0.0),
    }
    c1, c2 = curvature_sets.get(element_type, (weak_curvature, -weak_curvature))
    roc1 = 1 / c1 if c1 != 0 else 1e9
    roc2 = 1 / c2 if c2 != 0 else 1e9
    r_elem = foclen / (2 * fnum) * 1.1

    aperture_idx = next(
        (i for i, s in enumerate(surfaces) if s["type"] == "Aperture"), -1
    )

    if position == "front":
        insert_at = 0
    elif position == "rear":
        insert_at = len(surfaces)
    else:
        insert_at = aperture_idx if aperture_idx >= 0 else len(surfaces) // 2

    if insert_at == 0:
        d_start = 0.0
        for s in surfaces:
            s["d"] += thickness + 1.0
        d_next_back = surfaces[0]["d"] - (d_start + thickness) if surfaces else 1.0
    elif insert_at >= len(surfaces):
        last = surfaces[-1]
        d_start = last["d"] + last.get("d_next", 1.0)
        last["d_next"] = 1.0
        d_next_back = 5.0 * scale
        config["d_sensor"] += thickness + d_next_back
    else:
        prev = surfaces[insert_at - 1]
        d_start = prev["d"] + prev.get("d_next", 1.0)
        gap = prev["d_next"]
        prev["d_next"] = gap / 2
        d_next_back = gap / 2 - thickness
        if d_next_back < 0.5:
            d_next_back = 0.5
            config["d_sensor"] += thickness + 0.5
        for s in surfaces[insert_at:]:
            s["d"] += thickness + (gap / 2 - gap)

    max_idx = max(s["idx"] for s in surfaces) if surfaces else 0
    new_front = {
        "idx": max_idx + 1,
        "type": "Spheric",
        "r": r_elem,
        "c": c1,
        "roc": roc1,
        "d": d_start,
        "mat1": "air",
        "mat2": material,
        "d_next": thickness,
    }
    new_back = {
        "idx": max_idx + 2,
        "type": "Spheric",
        "r": r_elem,
        "c": c2,
        "roc": roc2,
        "d": d_start + thickness,
        "mat1": material,
        "mat2": "air",
        "d_next": d_next_back,
    }

    surfaces.insert(insert_at, new_front)
    surfaces.insert(insert_at + 1, new_back)
    for i, s in enumerate(surfaces):
        s["idx"] = i + 1

    n_elements = _count_elements(surfaces)
    new_id = lens_memory.store(config)
    return {
        "lens_id": new_id,
        "num_elements": n_elements,
        "config_summary": lens_memory.summarize(config),
        "summary": (
            f"Added {element_type} element ({material}) at {position} position. "
            f"Lens now has {n_elements} elements, {len(surfaces)} surfaces. "
            f"Use lens_id={new_id!r} for next steps."
        ),
    }


def remove_lens_element(lens_id: str, element_index: int) -> dict:
    """
    Remove a lens element (pair of surfaces) from a stored lens.

    Args:
        lens_id: ID of the current lens
        element_index: 1-based index of the element to remove

    Returns dict with new lens_id after removal.
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    config = copy.deepcopy(lens_config)
    surfaces = config["surfaces"]

    glass_elements = _find_glass_elements(surfaces)
    if element_index < 1 or element_index > len(glass_elements):
        return {
            "lens_id": lens_id,
            "error": (
                f"element_index {element_index} out of range "
                f"(lens has {len(glass_elements)} elements)"
            ),
        }

    start_surf_idx, end_surf_idx = glass_elements[element_index - 1]
    element_thickness = (
        surfaces[end_surf_idx]["d"] - surfaces[start_surf_idx]["d"]
        + surfaces[end_surf_idx].get("d_next", 0)
    )
    merged_d_next = element_thickness + surfaces[start_surf_idx - 1].get("d_next", 0) \
        if start_surf_idx > 0 else element_thickness

    if start_surf_idx > 0:
        surfaces[start_surf_idx - 1]["d_next"] = merged_d_next

    removed = surfaces[start_surf_idx : end_surf_idx + 1]
    shift = sum(s.get("d_next", 0) for s in removed[:-1]) + removed[0].get("d_next", 0)
    del surfaces[start_surf_idx : end_surf_idx + 1]
    for s in surfaces[start_surf_idx:]:
        s["d"] -= shift

    for i, s in enumerate(surfaces):
        s["idx"] = i + 1

    n_elements = _count_elements(surfaces)
    new_id = lens_memory.store(config)
    return {
        "lens_id": new_id,
        "num_elements": n_elements,
        "removed_element": element_index,
        "config_summary": lens_memory.summarize(config),
        "summary": (
            f"Removed element {element_index}. "
            f"Lens now has {n_elements} elements, {len(surfaces)} surfaces. "
            f"Use lens_id={new_id!r} for next steps."
        ),
    }


def save_lens(lens_id: str, filename: str = "final_lens", fmt: str = "json") -> dict:
    """
    Save a stored lens configuration to a file.

    Args:
        lens_id: ID of the lens to save
        filename: Output filename (without extension)
        fmt: 'json' or 'zmx'

    Returns dict with saved file path.
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    if not DEEPLENS_AVAILABLE and fmt == "zmx":
        return {"error": "ZMX export requires DeepLens installation.", "format": "zmx"}

    out_path = os.path.join(RESULTS_DIR, filename)

    if fmt == "json":
        json_path = out_path + ".json"
        with open(json_path, "w") as f:
            json.dump(lens_config, f, indent=2)
        return {"path": json_path, "format": "json", "summary": f"Saved to {json_path}"}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
        json.dump(lens_config, tmp)
        tmp_path = tmp.name
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lens = GeoLens(filename=tmp_path, device=device)
        zmx_path = out_path + ".zmx"
        lens.write_lens_zmx(zmx_path)
        return {"path": zmx_path, "format": "zmx", "summary": f"Saved to {zmx_path}"}
    except Exception as e:
        return {"error": str(e), "format": "zmx"}
    finally:
        os.unlink(tmp_path)


def generate_report(lens_id: str, metrics: dict) -> dict:
    """
    Generate a human-readable lens design report.

    Args:
        lens_id: ID of the final lens
        metrics: Evaluation metrics dict from evaluate_lens

    Returns dict with report text and key highlights.
    """
    try:
        lens_config = lens_memory.load(lens_id)
    except ValueError as e:
        return {"error": str(e), "summary": f"Failed to load lens: {e}"}

    foclen = lens_config.get("foclen", 0)
    fnum = lens_config.get("fnum", 0)
    sensor = lens_config.get("sensor_size", [0, 0])
    surfaces = lens_config.get("surfaces", [])
    n_elements = _count_elements(surfaces)

    rms = metrics.get("rms_spot_um", {})
    mtf = metrics.get("mtf_at_nyquist", "N/A")
    distortion = metrics.get("distortion_pct", "N/A")
    chroma = metrics.get("chromatic_aberration_um", "N/A")

    rms_str = ""
    if isinstance(rms, dict):
        parts = [f"  {k}: {v:.2f} µm" for k, v in rms.items()]
        rms_str = "\n".join(parts)
    elif isinstance(rms, (int, float)):
        rms_str = f"  RMS: {rms:.2f} µm"
    else:
        rms_str = f"  {rms}"

    report = f"""
╔══════════════════════════════════════════════════════╗
║           OPTICAL LENS DESIGN REPORT                 ║
╚══════════════════════════════════════════════════════╝

SPECIFICATIONS
  Focal Length   : {foclen:.2f} mm
  F-Number       : f/{fnum}
  Sensor Size    : {sensor[0]:.1f} × {sensor[1]:.1f} mm
  Elements       : {n_elements}
  Surfaces       : {len(surfaces)}
  Lens ID        : {lens_id}

ELEMENT STRUCTURE
{_format_element_table(surfaces)}

OPTICAL PERFORMANCE
  RMS Spot Size (µm):
{rms_str}

  MTF at Nyquist  : {mtf}
  Distortion      : {distortion}
  Chromatic Ab.   : {chroma}

ASSESSMENT
{_assessment(metrics)}
""".strip()

    return {
        "report": report,
        "focal_length": foclen,
        "fnum": fnum,
        "num_elements": n_elements,
        "rms_spot": rms,
        "overall_quality": _quality_score(metrics),
    }


# ─── Private helpers ───────────────────────────────────────────────────────────

def _extract_metrics(lens) -> dict:
    """Extract numerical metrics from a DeepLens GeoLens object."""
    import torch

    metrics: dict[str, Any] = {}

    try:
        rms_vals = {}
        for field_pct, label in [(0, "0%"), (0.7, "70%"), (1.0, "100%")]:
            loss = lens.loss_rms(
                num_grid=(5, 5),
                depth=-10000.0,
                num_rays=512,
                wvln=0.588,
            )
            rms_vals[label] = float(loss.item()) * 1000
        metrics["rms_spot_um"] = rms_vals
    except Exception as e:
        metrics["rms_spot_um"] = f"unavailable ({e})"

    try:
        distortion = lens.distortion(depth=-10000.0)
        metrics["distortion_pct"] = f"{float(distortion.item()) * 100:.2f}%"
    except Exception:
        metrics["distortion_pct"] = "N/A"

    try:
        from deeplens.basics import WAVE_RGB
        rms_r = float(lens.loss_rms(num_grid=(5, 5), depth=-10000.0,
                                     num_rays=256, wvln=WAVE_RGB[0]).item())
        rms_b = float(lens.loss_rms(num_grid=(5, 5), depth=-10000.0,
                                     num_rays=256, wvln=WAVE_RGB[2]).item())
        metrics["chromatic_aberration_um"] = f"{abs(rms_r - rms_b) * 1000:.2f} µm"
    except Exception:
        metrics["chromatic_aberration_um"] = "N/A"

    rms_center = metrics.get("rms_spot_um", {})
    if isinstance(rms_center, dict) and "0%" in rms_center:
        rms_val = rms_center["0%"]
        mtf_est = max(0, 1 - rms_val / 50)
        metrics["mtf_at_nyquist"] = f"~{mtf_est:.2f} (estimated)"
    else:
        metrics["mtf_at_nyquist"] = "N/A"

    return metrics


def _mock_optimization(original_id: str, lens_config: dict, iterations: int) -> dict:
    """Return a slightly perturbed config when DeepLens is unavailable."""
    config = copy.deepcopy(lens_config)
    for surf in config["surfaces"]:
        if surf["type"] != "Aperture" and surf.get("c", 0) != 0:
            surf["c"] *= 0.98
            if surf.get("roc"):
                surf["roc"] = 1 / surf["c"] if surf["c"] != 0 else 1e9
    new_id = lens_memory.store(config)
    return {
        "lens_id": new_id,
        "iterations_run": iterations,
        "status": "mock (DeepLens not installed)",
        "config_summary": lens_memory.summarize(config),
        "summary": (
            f"[MOCK] Optimization simulated for {iterations} iterations → lens_id={new_id!r}. "
            "Install DeepLens for real optimization."
        ),
    }


def _mock_evaluation(lens_config: dict) -> dict:
    """Return plausible fake metrics when DeepLens is unavailable."""
    foclen = lens_config.get("foclen", 50)
    n_surfaces = len([s for s in lens_config.get("surfaces", []) if s["type"] != "Aperture"])
    base_rms = max(2.0, 30.0 / n_surfaces)
    return {
        "status": "mock (DeepLens not installed)",
        "rms_spot_um": {
            "0% field": round(base_rms, 2),
            "70% field": round(base_rms * 1.8, 2),
            "100% field": round(base_rms * 3.2, 2),
        },
        "distortion_pct": f"{1.2 + foclen / 500:.2f}%",
        "chromatic_aberration_um": f"{base_rms * 0.6:.2f} µm",
        "mtf_at_nyquist": f"~{max(0, 1 - base_rms / 50):.2f} (estimated)",
        "summary": (
            "[MOCK] Metrics estimated. "
            "Install DeepLens for accurate simulation."
        ),
    }


def _count_elements(surfaces: list) -> int:
    """Count the number of distinct glass elements."""
    count = 0
    in_glass = False
    for s in surfaces:
        mat2 = s.get("mat2", "air")
        if mat2 != "air" and not in_glass:
            count += 1
            in_glass = True
        elif mat2 == "air":
            in_glass = False
    return count


def _find_glass_elements(surfaces: list) -> list:
    """Return list of (start_idx, end_idx) tuples for each glass element."""
    elements = []
    start = None
    for i, s in enumerate(surfaces):
        mat2 = s.get("mat2", "air")
        if mat2 != "air" and start is None:
            start = i
        elif mat2 == "air" and start is not None:
            elements.append((start, i))
            start = None
    return elements


def _format_element_table(surfaces: list) -> str:
    """Format a table of surfaces for the report."""
    lines = [f"  {'#':>3} {'Type':>10} {'R(mm)':>8} {'c(1/mm)':>10} {'Material':>15}"]
    lines.append("  " + "-" * 50)
    for s in surfaces:
        c = s.get("c", 0)
        r = s.get("r", 0)
        mat = s.get("mat2", "air") if s["type"] != "Aperture" else "—"
        lines.append(
            f"  {s['idx']:>3} {s['type']:>10} {r:>8.3f} {c:>10.5f} {mat:>15}"
        )
    return "\n".join(lines)


def _assessment(metrics: dict) -> str:
    """Generate a brief qualitative assessment."""
    rms = metrics.get("rms_spot_um", {})
    lines = []
    if isinstance(rms, dict):
        center = rms.get("0%", rms.get("0% field", 999))
        edge = rms.get("100%", rms.get("100% field", 999))
        if isinstance(center, (int, float)):
            if center < 5:
                lines.append("  ✓ Excellent center sharpness")
            elif center < 15:
                lines.append("  ~ Good center sharpness")
            else:
                lines.append("  ✗ Poor center sharpness — consider more elements")
        if isinstance(edge, (int, float)) and isinstance(center, (int, float)) and center > 0:
            ratio = edge / center
            if ratio < 2:
                lines.append("  ✓ Good field uniformity")
            elif ratio < 4:
                lines.append("  ~ Moderate field curvature")
            else:
                lines.append("  ✗ Strong field curvature — add field flattener")
    if not lines:
        lines.append("  Metrics insufficient for detailed assessment.")
    return "\n".join(lines)


def _quality_score(metrics: dict) -> str:
    """Return a simple quality label."""
    rms = metrics.get("rms_spot_um", {})
    if isinstance(rms, dict):
        center = rms.get("0%", rms.get("0% field", 999))
        if isinstance(center, (int, float)):
            if center < 5:
                return "Excellent"
            elif center < 15:
                return "Good"
            elif center < 30:
                return "Acceptable"
    return "Needs improvement"
