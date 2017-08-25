"""Construct and study PE Character Varieties."""
from .pecharvar import PECharVariety
from .complex_reps import PSL2CRepOf3ManifoldGroup
from .real_reps import PSL2RRepOf3ManifoldGroup
from .shape import ShapeSet, PolishedShapeSet
from .apoly import Apoly
from pe.sage_helper import _within_sage
if _within_sage:
    from .SL2R_lifting import SL2RLifter
import sys
