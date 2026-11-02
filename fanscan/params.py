"""Impeller parameter model shared by synth / analyze / cad.

All lengths are millimetres, all angles are degrees.

Blade angle convention (standard turbomachinery): beta is the angle between
the blade camber line and the local *circumferential* (tangential) direction.
0 < beta < 90 and the blade sweeps in the +theta direction as r increases
(see geometry.py).  Backward-curved impellers typically have beta2 ~ 20-50 deg.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ImpellerParams:
    # Primary dimensions
    outer_diameter: float            # D2, blade tip diameter
    inner_diameter: float            # D1, blade leading-edge (eye) diameter
    hub_diameter: float              # hub boss diameter (must be < D1)
    blade_count: int                 # Z
    blade_angle_inlet: float         # beta1 [deg], measured from tangential
    blade_angle_outlet: float        # beta2 [deg], measured from tangential
    blade_width: float               # b, axial blade height (constant for v0.1)
    blade_thickness: float           # t, uniform blade thickness
    # Secondary / structural
    backplate_thickness: float = 3.0
    hub_height: float | None = None  # defaults to blade_width + backplate
    shroud: bool = False             # add a front shroud (ring, or cone+eye ring if eye_diameter set)
    shroud_thickness: float = 2.0
    # ---- shrouded (production-type) impeller geometry, all optional -------
    eye_diameter: float | None = None        # shroud inlet ring inner diameter (defaults to D1)
    shroud_cone_height: float = 0.0          # axial rise of the shroud from the rim (z=b2) to the eye
    shroud_inlet_height: float = 0.0         # height of the cylindrical inlet ring above the cone top
    hub_bore_diameter: float = 0.0           # through-bore in the hub/backplate (0 = solid hub)
    blade_width_inlet: float | None = None   # b1 at the leading edge (defaults to blade_width = b2)
    # Free-form metadata (source image, confidence, notes...)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- derived ---------------------------------------------------------
    @property
    def r1(self) -> float:
        return self.inner_diameter / 2.0

    @property
    def r2(self) -> float:
        return self.outer_diameter / 2.0

    @property
    def r_hub(self) -> float:
        return self.hub_diameter / 2.0

    @property
    def r_eye(self) -> float:
        return (self.eye_diameter if self.eye_diameter else self.inner_diameter) / 2.0

    @property
    def b1(self) -> float:
        return self.blade_width_inlet if self.blade_width_inlet else self.blade_width

    @property
    def b2(self) -> float:
        return self.blade_width

    @property
    def is_shrouded(self) -> bool:
        return bool(self.shroud)

    @property
    def diameter_ratio(self) -> float:
        return self.inner_diameter / self.outer_diameter

    @property
    def blade_type(self) -> str:
        """backward / radial / forward classification from beta2."""
        if self.blade_angle_outlet < 85:
            return "backward"
        if self.blade_angle_outlet <= 95:
            return "radial"
        return "forward"

    # ---- validation ------------------------------------------------------
    def validate(self) -> None:
        if not (0 < self.hub_diameter < self.inner_diameter < self.outer_diameter):
            raise ValueError(
                f"need 0 < hub ({self.hub_diameter}) < D1 ({self.inner_diameter}) "
                f"< D2 ({self.outer_diameter})"
            )
        if self.blade_count < 2:
            raise ValueError("blade_count must be >= 2")
        for name in ("blade_angle_inlet", "blade_angle_outlet"):
            v = getattr(self, name)
            if not (1.0 <= v <= 179.0):
                raise ValueError(f"{name} must be within [1, 179] deg, got {v}")
        if self.blade_width <= 0 or self.blade_thickness <= 0:
            raise ValueError("blade_width and blade_thickness must be positive")
        if self.blade_thickness * 4 > (self.r2 - self.r1):
            raise ValueError("blade_thickness is implausibly large for the blade span")
        if self.eye_diameter is not None and not (self.hub_diameter < self.eye_diameter <= self.outer_diameter):
            raise ValueError("eye_diameter must lie between hub_diameter and outer_diameter")
        if self.hub_bore_diameter < 0 or self.hub_bore_diameter >= self.hub_diameter:
            raise ValueError("hub_bore_diameter must be >= 0 and < hub_diameter")
        if self.blade_width_inlet is not None and self.blade_width_inlet <= 0:
            raise ValueError("blade_width_inlet must be positive")
        if self.shroud_cone_height < 0 or self.shroud_inlet_height < 0:
            raise ValueError("shroud heights must be >= 0")

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_derived"] = {
            "blade_type": self.blade_type,
            "diameter_ratio": round(self.diameter_ratio, 4),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImpellerParams":
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ImpellerParams":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- comparison ------------------------------------------------------
    def compare(self, other: "ImpellerParams") -> dict[str, dict[str, float]]:
        """Relative / absolute error of *other* against self (ground truth)."""
        out: dict[str, dict[str, float]] = {}
        for name in (
            "outer_diameter", "inner_diameter", "hub_diameter",
            "blade_width", "blade_thickness",
        ):
            a, b = getattr(self, name), getattr(other, name)
            out[name] = {"truth": a, "est": b, "abs_err": b - a,
                         "rel_err_pct": 100.0 * (b - a) / a if a else float("nan")}
        for name in ("blade_angle_inlet", "blade_angle_outlet"):
            a, b = getattr(self, name), getattr(other, name)
            out[name] = {"truth": a, "est": b, "abs_err_deg": b - a}
        out["blade_count"] = {"truth": self.blade_count, "est": other.blade_count,
                              "abs_err": other.blade_count - self.blade_count}
        return out


# A plausible production-type shrouded impeller (modelled on the user's sample:
# cream ABS, conical front shroud with an inlet ring, bolted hub flange with a bore).
SHROUDED_PARAMS_FACTORY = lambda: ImpellerParams(  # noqa: E731
    outer_diameter=300.0,
    inner_diameter=190.0,
    hub_diameter=90.0,
    blade_count=11,
    blade_angle_inlet=30.0,
    blade_angle_outlet=40.0,
    blade_width=55.0,
    blade_width_inlet=75.0,
    blade_thickness=3.0,
    backplate_thickness=4.0,
    shroud=True,
    shroud_thickness=3.0,
    eye_diameter=180.0,
    shroud_cone_height=20.0,
    shroud_inlet_height=18.0,
    hub_bore_diameter=60.0,
)

# A plausible default: 200 mm backward-curved impeller.
DEFAULT_PARAMS = ImpellerParams(
    outer_diameter=200.0,
    inner_diameter=120.0,
    hub_diameter=40.0,
    blade_count=9,
    blade_angle_inlet=32.0,
    blade_angle_outlet=45.0,
    blade_width=40.0,
    blade_thickness=2.0,
)
