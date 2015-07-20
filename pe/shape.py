# -*- coding: utf-8 -*-
from snappy.snap.shapes import (pari, pari_column_vector, infinity_norm, pari_matrix,
                                pari_vector_to_list, enough_gluing_equations,
                                eval_gluing_equation, prec_bits_to_dec)
import numpy
from numpy import array, matrix, complex128, zeros, eye, transpose
from numpy.linalg import svd, norm
real_array = numpy.vectorize(float)
from .sage_helper import _within_sage

if _within_sage:
    from sage.all import ComplexField, pari
    def Number(z, precision=212):
        R = ComplexField(precision)
        return R(z)
else:
    from snappy.number import Number

def U1Q(p, q, precision=212):
    """An arbitrarily precise value for exp(2πip/q)"""
    result = (2*pari.pi(precision=precision)*p*pari('I')/q).exp(precision=precision)
    return Number(result, precision=precision)

def pari_set_precision(x, dec_prec):
    return pari(0) if x == 0 else pari(x).precision(dec_prec)

def pari_complex(z, dec_prec):
    try:
        real, imag = z.real(), z.imag()
    except TypeError:
        real, imag = z.real, z.imag
    return pari.complex(pari_set_precision(real, dec_prec), pari_set_precision(imag, dec_prec))

class GoodShapesNotFound(Exception):
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
        # if isinstance(manifold, snappy.ManifoldHP):
        #     self.hp_manifold = manifold
        # else:
        #     self.hp_manifold = manifold.high_precision()
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

    def dist(self, other):
        return norm(self.array - other.array)

    def __repr__(self):
        return ('<ShapeSet for %s:\n  '%self.manifold +
                '\n  '.join([repr(x) for x in self]) +
                '>')

    def update(self, values):
        self.array = array(values)

    def is_degenerate(self):
        moduli = abs(self.array)
        return ((moduli < 1.0E-6).any() or
                (moduli < 1.0E-6).any() or
                (moduli > 1.0E6).any())

    def SL2C(self, word):
        self.manifold.set_tetrahedra_shapes(self.array, None, [(0, 0)])
        G = self.manifold.fundamental_group()
        return G.SL2C(word)

    def O31(self, word):
        self.manifold.set_tetrahedra_shapes(self.array, None, [(0, 0)])
        G = self.manifold.fundamental_group()
        return G.O31(word)

    def has_real_traces(self):
        tolerance = self.reality_tolerance
        gens = self.manifold.fundamental_group().generators()
        gen_mats = [self.SL2C(g) for g in gens]
        for A in gen_mats:
            tr = complex(A[0, 0] + A[1, 1])
            if abs(tr.imag) > tolerance:
                return False
        mats = gen_mats[:]
        for _ in xrange(1, len(gens) + 1):
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
        gens = self.manifold.fundamental_group().generators()
        tolerance = self.SU2_tolerance
        # First check that all generators have real trace in [-2,2]
        for X in [self.SL2C(g) for g in gens]:
            tr = complex(X[0, 0] + X[1, 1])
            if abs(tr.imag) > tolerance:
                # print 'trace is not real'
                return False
            if abs(tr.real) > 2.0:
                # print 'trace is not in [-2,2]'
                return False
        # Get O31 matrix generators ...
        o31matrices = [real_array(array(self.O31(g))) for g in gens]
        # take the first two, ...
        A, B = o31matrices[:2]
        # find their axes, ...
        M = matrix(zeros((4, 4)))
        s, v = svd(A - eye(4))[1:]
        vt = transpose(v)
        M[:, [0, 1]] = vt[:, [n for n in range(4) if abs(s[n]) < tolerance]]
        s, v = svd(B - eye(4))[1:]
        vt = transpose(v)
        M[:, [2, 3]] = vt[:, [n for n in range(4) if abs(s[n]) < tolerance]]
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
        # Check if all of the generators fix the same point.
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
    >>> print '%.60s'%polished[0].real()
    0.9450156950804044984439848107085525618005253081486674978149
    >>> print '%.60s'%M.high_precision().tetrahedra_shapes('rect')[0].real()
    0.9450156950804044984439848107085525618005253081486674978149
    >>> M = snappy.Manifold('m071(7,0)')
    >>> beta = PolishedShapeSet(ShapeSet(M), U1Q(1,7, precision=256), precision=256)
    >>> print '%.60s'%beta[0].real()
    1.7806839263153037270854777557735393746612852691604941278274
    >>> print '%.60s'%M.high_precision().tetrahedra_shapes('rect')[0].real()
    1.7806839263153037270854777557735393746612852691604941278274
    >>> beta.advance_holonomy(1, 128)
    >>> print '%.60s'%beta[0].real()
    1.8594892443933700576759347698861791174458226666979385882887
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
        precision = self._precision
        manifold = self.manifold
        dec_prec = prec_bits_to_dec(precision)
        working_prec = dec_prec + 10
        target_espilon = pari_set_precision(10.0, working_prec)**-dec_prec
        det_epsilon = pari_set_precision(10.0, working_prec)**-(dec_prec//10)
        init_shapes = pari_column_vector(
            [pari_complex(z, working_prec) for z in init_shapes])
        init_equations = manifold.gluing_equations('rect')
        target = pari_complex(self.target_holonomy, dec_prec)
        error = self._gluing_equation_error(init_equations, init_shapes, target)
        if flag_initial_error and error > pari(0.000001):
            raise GoodShapesNotFound('Initial solution not very good')

        # Now begin the actual computation

        eqns = enough_gluing_equations(manifold)
        assert eqns[-1] == manifold.gluing_equations('rect')[-1]

        shapes = init_shapes
        for i in range(100):
            errors = self._gluing_equation_errors(eqns, shapes, target)
            if infinity_norm(errors) < target_espilon:
                break
            shape_list = pari_vector_to_list(shapes)
            derivative = [[eqn[0][i]/z  - eqn[1][i]/(1 - z)
                           for i, z in enumerate(shape_list)] for eqn in eqns]
            derivative[-1] = [target*x for x in derivative[-1]]
            derivative = pari_matrix(derivative)

            det = derivative.matdet().abs()
            if min(det, 1/det) < det_epsilon:
                break  # Pari might crash
            gauss = derivative.matsolve(pari_column_vector(errors))
            shapes = shapes - gauss

        # Check to make sure things worked out ok.
        error = self._gluing_equation_error(init_equations, shapes, target)
        total_change = infinity_norm(init_shapes - shapes)
        if error > 1000*target_espilon:
            raise GoodShapesNotFound('Failed to find solution')
        if flag_initial_error and total_change > pari(0.0000001):
            raise GoodShapesNotFound('Moved to far finding a good solution')
        shapes = pari_vector_to_list(shapes)
        self.shapelist = [Number(z, precision=precision) for z in shapes]

    def __getitem__(self, index):
        return self.shapelist[index]

    @staticmethod
    def _gluing_equation_errors(eqns, shapes, RHS_of_last_eqn):
        last = [eval_gluing_equation(eqns[-1], shapes) - RHS_of_last_eqn]
        return [eval_gluing_equation(eqn, shapes) - 1
                for eqn in eqns[:-1]] + last

    def _gluing_equation_error(self, eqns, shapes, RHS_of_last_eqn):
        return infinity_norm(self._gluing_equation_errors(
            eqns, shapes, RHS_of_last_eqn))

    def precision(self):
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

