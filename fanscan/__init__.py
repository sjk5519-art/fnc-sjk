"""fanscan — centrifugal fan impeller shape analysis from photographs.

Pipeline: photo (axial view) -> ImpellerParams (JSON) -> STEP (CadQuery).
"""
from .params import ImpellerParams

__all__ = ["ImpellerParams"]
__version__ = "0.1.0"
