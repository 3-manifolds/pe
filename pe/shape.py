# -*- coding: utf-8 -*-
"""
Define the classes ShapeSet and PolishedShapeSet, which represent a tuple of
solutions to a gluing system.  The shapes in a ShapeSet are double precision
complex numbers, while those in a PolishedShapeSet have arbitrary precision.
"""

from __future__ import print_function
from snappy.snap.shapes import (enough_gluing_equations, infinity_norm,
                                gluing_equation_errors, eval_gluing_equation)
from numpy import array, matrix, complex128, zeros, eye, transpose, vectorize
from numpy.linalg import svd, norm
real_array = vectorize(float)
from .sage_helper import _within_sage

if _within_sage:
    from sage.all import ComplexField, Matrix, pari
    from sage.libs.mpmath.utils import mpmath_to_sage
    import mpmath
    def Number(z, precision=212):
        """In sage we use Sage numbers."""
        R = ComplexField(precision)
        if isinstance(z, mpmath.mpc):
            z = mpmath_to_sage(z, precision)
        return R(z)
else:
    from snappy.number import Number

def U1Q(p, q, precision=212):
    """An arbitrarily precise value for exp(2πip/q)"""
    result = (2*pari.pi(precision=precision)*p*pari('I')/q).exp(
        precision=precision)
    return Number(result, precision=precision)

def pari_set_precision(x, precision):
    """
    Promote a real number to an arbitrary precision Pari number with the same
    value.  The precision should be given in bits.
    """
    return pari(0) if x == 0 else pari(x).precision(prec_bits_to_dec(precision))

def pari_complex(z, precision):
    """
    Promote a complex number to an arbitrary precision Pari number with the same
    value.  The precision should be given in bits.
    """
    try:
        real, imag = z.real(), z.imag()
    except TypeError:
        real, imag = z.real, z.imag
    return pari.complex(pari_set_precision(real, precision),
                        pari_set_precision(imag, precision))

class GoodShapesNotFound(Exception):
    """Exception generated when precise shapes cannot be found."""
    pass

class ShapeSet(object):
    """
    A vector of shape parameters, stored as a numpy.array.
    Many methods are available: ...
    Instantiate with a sequence of complex numbers.
    """
    SU2_tolerance = 1.0E-5
    reality_tolerance = 1.0E-10

    def __init__(self, manifold, values=None):
        self.manifold = manifold
        if values is None:
            values = [complex128(z) for z in manifold.tetrahedra_shapes('rect')]
        self.array = array(values)

    def __str__(self):
        return self.array.__str__()

    def __getitem__(self, value):
        return self.array[value]

    def __len__(self):
        return len(self.array)

    def __eq__(self, other):
        return norm(self.array - other.array) < 1.0E-6

    def __repr__(self):
        return ('<ShapeSet for %s:\n  '%self.manifold +
                '\n  '.join([repr(x) for x in self]) +
                '>')

    def dist(self, other):
        """
        Return the L^2 distance from this shapeset to the other.
        """
        return norm(self.array - other.array)

    def update(self, values):
        """Replace the shape values with new ones."""
        self.array = array(values)

    def is_degenerate(self):
        """True if any shape in this ShapeSet is degenerate."""
        moduli = abs(self.array)
        dist_to_1 = abs(self.array - 1.0)
        return ((moduli < 1.0E-6).any() or
                (dist_to_1 < 1.0E-6).any() or
                (moduli > 1.0E6).any())
    

    def _SL2C(self, word):
        self.manifold.set_tetrahedra_shapes(self.array, None, [(0, 0)])
        G = self.manifold.fundamental_group()
        return G.SL2C(word)

    def _O31(self, word):
        self.manifold.set_tetrahedra_shapes(self.array, None, [(0, 0)])
        G = self.manifold.fundamental_group()
        return G.O31(word)

    def has_real_traces(self):
        """True if the holonomy rep associated to these shapes has a real character"""
        tolerance = self.reality_tolerance
        gens = self.manifold.fundamental_group().generators()
        gen_mats = [self._SL2C(g) for g in gens]
        for A in gen_mats:
            tr = complex(A[0, 0] + A[1, 1])
            if abs(tr.imag) > tolerance:
                return False
        mats = gen_mats[:]
        for _ in range(1, len(gens) + 1):
            new_mats = []
            for A in gen_mats:
                for B in mats:
                    C = B*A
                    tr = complex(C[0, 0] + C[1, 1])
                    if abs(tr.imag) > tolerance:
                        return False
                    new_mats.append(C)
        return True

    def in_SU2(self):
        """True if the holonomy rep associated to these shapes has image in  SU(2)."""
        gens = self.manifold.fundamental_group().generators()
        tolerance = self.SU2_tolerance
        # First check that all generators have real trace in (-2,2)
        # and look for non-trivial generators
        good_gens = []
        for g in gens:
            X = self._SL2C(g)
            tr = complex(X[0, 0] + X[1, 1])
            if abs(tr.imag) > tolerance:
                # print 'trace is not real'
                return False
            if abs(tr.real) > 2.0:
                # print 'trace is not in [-2,2]'
                return False
            if abs(tr.real) < 2.0 - tolerance:
                good_gens.append(g)
        if len(good_gens) < 2:
            return True
            #raise RuntimeError('Yikes! This rep is abelian!')
        # Get the first two non-trivial O31 matrix generators ...
        A, B = [real_array(array(self._O31(g))) for g in good_gens[:2]]
        # find their axes, ...
        M = matrix(zeros((4, 4)))
        try:
            s, v = svd(A - eye(4))[1:]
            vt = transpose(v)
            M[:, [0, 1]] = vt[:, [n for n in range(4) if abs(s[n]) < tolerance]]
            s, v = svd(B - eye(4))[1:]
            vt = transpose(v)
            M[:, [2, 3]] = vt[:, [n for n in range(4) if abs(s[n]) < tolerance]]
        except:
            raise RuntimeError('Failed to find two crossing axes.')
        # check if the axes cross,
        # and find the fixed point (i.e. Minkwoski line)
        s, v = svd(M)[1:]
        vt = transpose(v)
        rel = vt[:, [n for n in range(4) if abs(s[n]) < tolerance]]
        if rel.shape != (4, 1):
            # print 'linear algebra failure'
            return False
        # We now have two descriptions -- let's average them.
        rel[2] = -rel[2]
        rel[3] = -rel[3]
        fix = M*rel
        # Check if the fixed line is in the light cone.
        if abs(fix[0]) <= norm(fix[1:]):
            # print 'fixed line is not in the light cone'
            return False
        # Check that all of the generators fix the same point.
        o31matrices = [real_array(array(self._O31(g))) for g in gens]
        for O in o31matrices:
            if norm(O*fix - fix) > tolerance:
                # print 'some generators do not share the fixed point.'
                return False
        return True

class PolishedShapeSet(object):
    """An arbitrarily precise solution to the gluing equations with a
    specified target value for the meridian holonomy.  Initialize with
    a rough ShapeSet object and a target_holonomy.

    >>> import snappy
    >>> M = snappy.Manifold('m071(0,0)')
    >>> rough = ShapeSet(M)
    >>> rough[0]
    (0.94501569508040595+1.0738656547982881j)
    >>> polished = PolishedShapeSet(rough, target_holonomy=1.0)
    >>> print '%.55s'%polished[0].real()
    0.94501569508040449844398481070855256180052530814866749
    >>> print '%.55s'%M.high_precision().tetrahedra_shapes('rect')[0].real()
    0.94501569508040449844398481070855256180052530814866749
    >>> M = snappy.Manifold('m071(7,0)')
    >>> beta = PolishedShapeSet(ShapeSet(M), U1Q(1,7, precision=256), precision=256)
    >>> print '%.55s'%beta[0].real()
    1.78068392631530372708547775577353937466128526916049412
    >>> print '%.55s'%M.high_precision().tetrahedra_shapes('rect')[0].real()
    1.78068392631530372708547775577353937466128526916049412
    >>> beta.advance_holonomy(1, 128)
    >>> print '%.55s'%beta[0].real()
    1.85948924439337005767593476988617911744582266669793858
    """
    def __init__(self, rough_shapes, target_holonomy, precision=212):
        self.rough_shapes = rough_shapes
        self.target_holonomy = target_holonomy
        self._precision = precision
        self.manifold = rough_shapes.manifold.copy()
        self.manifold.dehn_fill((1, 0))
        self.polish(init_shapes=rough_shapes.array)

    def __repr__(self):
        return ('<ShapeSet for %s:\n  '%self.manifold +
                '\n  '.join([repr(x) for x in self]) +
                '>')

    def polish(self, init_shapes, flag_initial_error=True):
        """Use Newton's method to compute precise shapes from rough ones."""
        precision = self._precision
        manifold = self.manifold
        #working_prec = precision + 32
        working_prec = precision + 64
        mpmath.mp.prec = working_prec 
        target_epsilon = mpmath.mpmathify(2.0)**-precision
        det_epsilon = mpmath.mpmathify(2.0)**(32 - precision)
        #shapes = [mpmath.mpmathify(z) for z in init_shapes]
        shapes = mpmath.matrix(init_shapes)
        init_equations = manifold.gluing_equations('rect')
        target = mpmath.mpmathify(self.target_holonomy)
        error = self._gluing_equation_error(init_equations, shapes, target)
        if flag_initial_error and error > 0.000001:
            raise GoodShapesNotFound('Initial solution not very good: error=%s'%error)

        # Now begin the actual computation

        eqns = enough_gluing_equations(manifold)
        assert eqns[-1] == manifold.gluing_equations('rect')[-1]
        for i in range(100):
            errors = self._gluing_equation_errors(eqns, shapes, target)
            if infinity_norm(errors) < target_epsilon:
                break
            derivative = [[int(eqn[0][i])/z  - int(eqn[1][i])/(1 - z)
                           for i, z in enumerate(shapes)] for eqn in eqns]
            derivative[-1] = [target*x for x in derivative[-1]]
            derivative = mpmath.matrix(derivative)
            #det = derivative.matdet().abs()
            det = abs(mpmath.det(derivative))
            if min(det, 1/det) < det_epsilon:
                raise GoodShapesNotFound('Gluing system is too singular (|det| = %s).'%det)
            delta = mpmath.lu_solve(derivative, errors)
            shapes = shapes - delta

        # Check to make sure things worked out ok.
        error = self._gluing_equation_error(init_equations, shapes, target)
        #total_change = infinity_norm(init_shapes - shapes)
        if error > 1000*target_epsilon:
            raise GoodShapesNotFound('Failed to find solution')
        #if flag_initial_error and total_change > pari(0.0000001):
        #    raise GoodShapesNotFound('Moved too far finding a good solution')
        self.shapelist = [Number(z, precision=precision) for z in shapes]

    def __getitem__(self, index):
        return self.shapelist[index]

    @staticmethod
    def _gluing_equation_errors(eqns, shapes, RHS_of_last_eqn):
        last = [eval_gluing_equation(eqns[-1], shapes) - RHS_of_last_eqn]
        result = []
        for eqn in eqns[:-1]:
            val = eval_gluing_equation(eqn, shapes)
            result.append( val - 1)
        return result + last

    def _gluing_equation_error(self, eqns, shapes, RHS_of_last_eqn):
        return infinity_norm(self._gluing_equation_errors(
            eqns, shapes, RHS_of_last_eqn))

    def precision(self):
        """Return the precision used for these polished shapes."""
        return self._precision

    def rep_type(self):
        """
        Return the type of the holonomy representation determined by these
        shapes.
        """
        if self.rough_shapes.in_SU2():
            return 'SU(2) rep'
        elif self.rough_shapes.has_real_traces():
            return 'SL(2,R) rep'
        elif (self.target_holonomy.abs() -1).abs() < 2**(0.8*self._precision):
            return 'Non-real PE rep'
        else:
            return 'generic rep'

    def advance_holonomy(self, p, q):
        """Change the target_holonomy by exp(2πip/q)"""
        z = U1Q(p, q, self._precision)
        self.target_holonomy = z*self.target_holonomy
        self.polish(self.shapelist, False)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

