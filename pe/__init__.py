"""Construct and study PE Character Varieties."""
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from .pecharvar import PECharVariety
from .prcharvar import PRCharVariety
from .elevation import CircleElevation, LineElevation
from .complex_reps import PSL2CRepOf3ManifoldGroup
from .real_reps import PSL2RRepOf3ManifoldGroup
from .shape import ShapeSet, PolishedShapeSet
from .apoly import Apoly, ComputedApoly
from pe.sage_helper import _within_sage
if _within_sage:
    from .SL2R_lifting import SL2RLifter

