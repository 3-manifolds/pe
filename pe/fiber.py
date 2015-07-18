from numpy import complex128
from .gluing import GluingSystem
from .shape import Shapes, PolishedShapes

class Fiber(object):
    """A fiber for the rational function [holonomy of the meridian]
    restricted to the curve defined by the gluing system for a
    triangulated cusped manifold.  Can be initialized with a PHCSystem
    and a list of PHCsolutions.

    """
    def __init__(self, manifold, H_meridian, gluing_system=None,
                 PHCsystem=None, shapes=None, tolerance=1.0E-05):
        self.manifold = manifold
        # Here the tolerance is used to determine which of the PHC solutions
        # are at infinity.
        self.H_meridian = H_meridian
        self.tolerance = tolerance
        if shapes:
            self.shapes = [Shapes(self.manifold, S) for S in shapes]
        if gluing_system is None:
            self.gluing_system = GluingSystem(manifold)
        else:
            self.gluing_system = gluing_system
        self.system = PHCsystem
        if self.system:
            N = self.system.num_variables()/2
            self.solutions = self.system.solution_list(tolerance=self.tolerance)
            # We only keep the "X" variables.
            self.shapes = [Shapes(self.manifold, S.point[:N]) for S in self.solutions]

    def __repr__(self):
        return "Fiber(ManifoldHP('%s'),\n%s,\nshapes=%s\n)"%(
            repr(self.manifold),
            repr(self.H_meridian),
            repr([list(x) for x in self.shapes]).replace('],', '],\n')
            )

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        return self.shapes[index]

    def __eq__(self, other):
        """
        This ignores multiplicities.
        """
        for p in self.shapes:
            if p not in other.shapes:
                return False
        for p in other.shapes:
            if p not in self.shapes:
                return False
        return True

    def collision(self):
        for n, p in enumerate(self.shapes):
            for q in self.shapes[n+1:]:
                if p.dist(q) < 1.0E-10:
                    return True
        return False

    def is_finite(self):
        """
        Check if any cross-ratios are 0 or 1
        """
        for p in self.shapes:
            if p.is_degenerate():
                return False
        return True

    def phc_details(self):
        # broken if not instantiated with a PHCsystem
        for n, s in enumerate(self.solutions):
            print 'solution #%s:'%n
            print s

    def phc_residuals(self):
        # broken if not instantiated with a PHCsystem
        for n, s in enumerate(self.solutions):
            print n, s.res

    def polish(self):
        polished = self.polished_shapelist(precision=96)
        for shapes, polished_shapes in zip(self, polished):
            shapes.update([complex128(z) for z in polished_shapes])

    def phc_Tillmann_points(self):
        # broken if not instantiated with a PHCsystem
        if self.system is None:
            return []
        result = []
        for n, s in enumerate(self.solutions):
            if s.t != 1.0 or self.shapes[n].is_degenerate():
                result.append(n)
        return result

    def permutation(self, other):
        """
        return a list of pairs (m, n) where self.shapes[m] is
        closest to other.shapes[n].
        """
        result = []
        other_shapes = other.shapes
        remaining = set(range(len(other_shapes)))
        for m, shape in enumerate(self.shapes):
            dist_to_shape = lambda k, s=shape: s.dist(other_shapes[k])
            n = min(remaining, key=dist_to_shape)
            result.append((m, n))
            remaining.remove(n)
        return result

    def transport(self, target_holonomy, debug=False):
        """
        Transport this fiber to a different target holonomy.
        """
        shapes = []
        dT = 1.0
        while True:
            if dT < 1.0/64:
                raise ValueError('Collision unavoidable. Try a different radius.')
            for shape in self.shapes:
                Zn = self.gluing_system.track(shape.array,
                                              target_holonomy,
                                              dT=dT,
                                              debug=debug)
                shapes.append(Zn)
            result = Fiber(self.manifold, target_holonomy,
                           gluing_system=self.gluing_system,
                           shapes=shapes)
            if result.collision():
                dT *= 0.5
            else:
                break
        return result

    def polished_shape(self, n, target_holonomy=None, dec_prec=None, precision=200):
        if target_holonomy is None:
            target_holonomy = self.H_meridian
        return PolishedShapes(self[n], target_holonomy, dec_prec=dec_prec, precision=precision)

    def polished_shapelist(self, target_holonomy=None, dec_prec=None, precision=200):
        if target_holonomy is None:
            target_holonomy = self.H_meridian
        return [PolishedShapes(S, target_holonomy, dec_prec, precision) for S in self]

