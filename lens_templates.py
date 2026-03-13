"""
Lens template selection logic.
Chooses an appropriate starting lens structure based on optical specifications.
"""

import json
import math
import os
from typing import Optional

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def select_template(
    focal_length: float,
    fnum: float,
    fov_deg: float,
    sensor_size: Optional[tuple] = None,
    num_elements: Optional[int] = None,
) -> dict:
    """
    Select and scale an appropriate lens template based on specifications.

    Args:
        focal_length: Target focal length in mm
        fnum: Target f-number (e.g., 2.8)
        fov_deg: Full diagonal field of view in degrees
        sensor_size: (width, height) in mm, defaults based on focal_length
        num_elements: Explicit number of lens elements (overrides auto-selection)

    Returns:
        Lens configuration dict (DeepLens JSON format)
    """
    template_name = _choose_template(focal_length, fnum, fov_deg, num_elements)
    config = _load_template(template_name)
    config = _scale_template(config, focal_length, fnum, fov_deg, sensor_size)
    config["info"] = (
        f"Auto-generated {template_name} design: "
        f"{focal_length:.0f}mm f/{fnum} {fov_deg:.0f}deg FoV"
    )
    return config


def _choose_template(
    focal_length: float,
    fnum: float,
    fov_deg: float,
    num_elements: Optional[int],
) -> str:
    """Rule-based template selection."""
    if num_elements is not None:
        if num_elements == 1:
            return "singlet"
        elif num_elements == 2:
            return "doublet"
        elif num_elements == 3:
            return "triplet"
        elif num_elements <= 6:
            return "double_gauss"

    # Wide-angle: FOV > 70 degrees
    if fov_deg > 70:
        return "wide_angle"

    # Telephoto: effective focal length much longer than sensor diagonal
    # Standard definition: telephoto ratio < 0.8
    if focal_length >= 135:
        return "telephoto"

    # Large aperture (f/2 or faster) standard range → Double Gauss
    if fnum <= 2.0 and fov_deg < 60:
        return "double_gauss"

    # Moderate FOV (40-70°) with moderate aperture → Triplet (Cooke)
    if 35 <= fov_deg <= 70 and fnum > 2.0:
        return "triplet"

    # Slow lenses (f/5.6+) with small FOV → Singlet or Doublet
    if fnum >= 8.0 and fov_deg < 20:
        return "singlet"

    if fnum >= 4.0 and fov_deg < 30:
        return "doublet"

    # Default: triplet is a good general-purpose starting point
    return "triplet"


def _load_template(name: str) -> dict:
    """Load a template JSON file."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _scale_template(
    config: dict,
    target_foclen: float,
    target_fnum: float,
    fov_deg: float,
    sensor_size: Optional[tuple],
) -> dict:
    """
    Scale a template to match the target focal length and sensor size.
    Uses linear scaling for all distances and curvatures.
    """
    import copy
    config = copy.deepcopy(config)

    ref_foclen = config["foclen"]
    scale = target_foclen / ref_foclen

    # Scale all surface distances and curvatures
    for surf in config["surfaces"]:
        surf["r"] = surf["r"] * scale
        surf["c"] = surf["c"] / scale  # curvature = 1/ROC, so scales inversely
        if surf.get("roc") and surf["roc"] != 0:
            surf["roc"] = surf["roc"] * scale
        surf["d"] = surf["d"] * scale
        surf["d_next"] = surf["d_next"] * scale
        # Scale aspheric coefficients (ai_n scales as 1/scale^(n-1))
        for n, key in enumerate(["ai2", "ai4", "ai6", "ai8", "ai10", "ai12"], start=1):
            if key in surf:
                surf[key] = surf[key] / (scale ** (2 * n - 1))

    # Update global parameters
    config["foclen"] = target_foclen
    config["fnum"] = target_fnum
    config["d_sensor"] = config["d_sensor"] * scale

    # Update aperture radius based on f-number
    entrance_pupil_r = target_foclen / (2 * target_fnum)
    # Scale aperture stop radius
    for surf in config["surfaces"]:
        if surf["type"] == "Aperture":
            surf["r"] = entrance_pupil_r

    # Update sensor size based on FOV and focal length
    if sensor_size is not None:
        w, h = sensor_size
        config["sensor_size"] = [w, h]
        config["r_sensor"] = math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    else:
        # Compute sensor size from FOV and focal length
        # half-diagonal = f * tan(FOV/2)
        half_diag = target_foclen * math.tan(math.radians(fov_deg / 2))
        config["r_sensor"] = half_diag
        # Assume 3:2 aspect ratio
        side = half_diag * math.sqrt(2) * (2 / math.sqrt(13))
        config["sensor_size"] = [side * 1.5, side]

    # Re-index surfaces
    for i, surf in enumerate(config["surfaces"]):
        surf["idx"] = i + 1

    return config


def get_template_info() -> dict:
    """Return info about all available templates."""
    templates = {}
    for fname in os.listdir(TEMPLATE_DIR):
        if fname.endswith(".json"):
            name = fname[:-5]
            try:
                cfg = _load_template(name)
                templates[name] = {
                    "focal_length": cfg["foclen"],
                    "fnum": cfg["fnum"],
                    "num_surfaces": len(cfg["surfaces"]),
                    "info": cfg.get("info", ""),
                }
            except Exception as e:
                templates[name] = {"error": str(e)}
    return templates
