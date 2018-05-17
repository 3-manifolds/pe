# -*- coding: utf-8 -*-

"""
Define the Fiber class.

A Fiber object represents the pre-image of a point under the meridian
holonomy map on the gluing variety.
"""

from __future__ import print_function
from numpy import complex128
from .shape import ShapeSet, PolishedShapeSet, GoodShapesNotFound

class Fiber(object):
    """
    In the typical case, where each component of the gluing variety of
    a given triangulated cusped manifold is a curve on which the
    holonomy of the meridian is non-constant, A Fiber object
    represents a fiber for the meridian holonomy as a rational map
    from the gluing variety to C.
    """

    def __init__(self, manifold, H_meridian, PHCsystem=None, shapes=None,
                 tolerance=1.0E-06):
        # The tolerance is used to determine which of the PHC solutions
        # are at infinity.
        self.manifold = manifold
        self.H_meridian = H_meridian
        self.tolerance = tolerance
        self.system = PHCsystem
        if shapes:
            self.shapes = [ShapeSet(manifold, S) for S in shapes]
        elif self.system:
            N = self.system.num_variables()/2
            self.solutions = self.system.solution_list(tolerance=self.tolerance)
            # We only keep the "X" variables.
            self.shapes = [ShapeSet(self.manifold, S.point[:N]) for S in self.solutions]
        else:
            raise ValueError('Fiber object requires a PHC system or a list of shapes.')

    def __str__(self):
        return "Fiber(Manifold('%s'),\n%s,\nshapes=%s\n)"%(
            repr(self.manifold),
            repr(self.H_meridian),
            repr([list(x) for x in self.shapes]).replace('],', '],\n')
            )

    def __repr__(self):
        return '<Fiber for %s over %s>'%(self.manifold, self.H_meridian)

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

    def _random_vector(self):
        return [exp(2*random()*pi*1j) for n in xrange(self.num_shapes)]

    def clean(self, coranks):
        """
        This should only be called for the random base fiber.

        Given a list of coranks of the gluing system at each of our shapes, we
        could try to augment the shapeset with a set of enough random extra
        linear equations to cut the dimension of the component containing that
        shape point down to 1.

        For now, though, we just raise an exception if high-dimensional components
        are found.
        """
        if set(coranks) != set([0]):
            raise RuntimeError("Found high dimensional components of the gluing variety.")

    def collision(self):
        """
        Does this fiber contain a point of multiplicity > 1?  This will occur when
        the meridian holonomy has a singular point on the circle elevation, or when
        the gluing variety has a singuarity on the circle elevation.  But those things
        should not occur if the radius has been chosen generically.  The other cause
        for a collision is that Newton's method converges to the same shapeset when
        startied at two distinct points of the previously computed fiber.
        """
        for n, p in enumerate(self.shapes):
            for m, q in enumerate(self.shapes[n+1:]):
                if p.dist(q) < 1.0E-10:
                    print("\nCollision of shapesets %s and %s at %s."%(
                        n, n+m+1, self.H_meridian))
                    return True
        return False

    def match_to(self, other):
        """
        Sort the points of this fiber so that the nth point of this fiber is close to
        the nth point of the other fiber.  (Useful if this fiber is computed with PHC
        after a failed transport.)
        """
        other_shapes = list(enumerate(other.shapes))
        result = list(range(len(self)))

        for s in self.shapes:
            distance = float('inf')
            nearest = -1
            for m, S in enumerate(other_shapes):
                next_distance = s.dist(S[1])
                if next_distance < distance:
                    nearest, nearest_index = S[0], m
                    distance = next_distance
            result[nearest] = s
            other_shapes.pop(nearest_index)
        self.shapes = result

    def is_finite(self):
        """
        Check if any cross-ratios are 0 or 1
        """
        for p in self.shapes:
            if p.is_degenerate():
                return False
        return True

    def phc_details(self):
        """Print all shapes.  Only works for fibers constructed with PHC."""
        for n, s in enumerate(self.solutions):
            print('solution #%s:'%n)
            print(s)

    def phc_residuals(self):
        """Print the residuals for the PHC approximations."""
        for n, s in enumerate(self.solutions):
            print(n, s.res)

    def polish(self):
        """Ensure that the shapes are accurate to full double precision."""
        precision = 128
        polished = None
        for _ in range(4):
            try:
                polished = self.polished_shapelist(precision=precision)
                break
            except GoodShapesNotFound:
                precision *= 2
        if polished is None:
            raise GoodShapesNotFound('Failed to polish shapeset')
        for shapes, polished_shapes in zip(self, polished):
            shapes.update([complex128(z) for z in polished_shapes])

    def phc_Tillmann_points(self):
        """Return the solutions which contains degenerate shapes."""
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
        Return a list of pairs (m, n) where self.shapes[m] is
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

    def polished_shapelist(self, target_holonomy=None, precision=200):
        """Compute all shapes to arbitrary binary precision (default 200)."""
        if target_holonomy is None:
            target_holonomy = self.H_meridian
        return [PolishedShapeSet(S, target_holonomy, precision) for S in self]

