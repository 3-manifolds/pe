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
from .version import version as __version__
from pe.sage_helper import _within_sage
from snappy import Manifold, ManifoldHP
if _within_sage:
    from .SL2R_lifting import EllipticSL2RLifter, HyperbolicSL2RLifter
