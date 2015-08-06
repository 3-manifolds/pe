# -*- coding: utf-8 -*-
"""
Define the class PSL2RRepOf3ManifoldGroup which represents an arbitrary precision
holonomy representation with image in SL(2,R).
"""

from .sage_helper import _within_sage, get_pi
from .complex_reps import (PSL2CRepOf3ManifoldGroup, polished_group,
                           apply_representation, GL2C_inverse, SL2C_inverse,
                           CheckRepresentationFailed, conjugacy_classes_in_Fn)
from pe.euler import orientation, PSL2RtildeElement, LiftedFreeGroupRep

if _within_sage:
    from sage.all import RealField, MatrixSpace, ZZ, vector, matrix, pari, arg
    eigenvalues = lambda A: A.charpoly().roots(A.base_ring(), False)
    Id2 = MatrixSpace(ZZ, 2)(1)
    complex_I = lambda R: R.gen()
    complex_field = lambda R: R.complex_field()
else:
    from cypari.gen import pari
    from snappy.number import Number, SnapPyNumbers
    from snappy.snap.utilities import Vector2 as vector, Matrix2x2 as matrix
    eigenvalues = lambda A: A.eigenvalues()
    Id2 = matrix(1, 0, 0, 1)
    RealField = SnapPyNumbers
    ComplexField = SnapPyNumbers
    complex_I = lambda R: R.I()
    complex_field = lambda R: R
    def arg(x):
        """Use the object's arg method."""
        if isinstance(x, Number):
            return x.arg()
        else:
            return Number(x).arg()

class CouldNotConjugateIntoPSL2R(Exception):
    """Exception generated when a representation cannot be conjugated into PSL(2,R)."""
    pass

def clean_real(r):
    """Make essentially zero numbers really be zero."""
    RR = r.parent()
    epsilon = RR(2.0)**(-0.5*RR.precision())
    return RR(0) if abs(r) < epsilon else r

def real_part_of_matrix_with_error(A):
    """
    Take the real part and return the size of the imaginary part.

    This matrix is assumed to be representing an element of PSL(2,R) and
    here we enforce the convention that elements of PSL(2,R) should be
    represented by matrices with non-negative trace.
    """
    RR = RealField(A.base_ring().precision())
    entries = A.list()
    real_parts = [clean_real(x.real()) for x in entries]
    error = max([abs(x.imag()) for x in entries])
    B = matrix(RR, [[real_parts[0], real_parts[1]], [real_parts[2], real_parts[3]]])
    if B.trace() < 0:
        B = -B
    return B, error

def real_part_of_matrices_with_error(matrices):
    """Return a list of real parts of matrices in a list."""
    real_with_errors = [real_part_of_matrix_with_error(A) for A in matrices]
    return [r for r, _ in real_with_errors], max(e for r, e in real_with_errors)

def normalize_vector(v):
    """Divide this non-zero vector by its L2 norm."""
    return v/v.norm()

def apply_matrix(mat, v):
    """Multiply the matrix times the vector and return the normalized result."""
    return normalize_vector(mat*v)

def vector_dist(a, b):
    """Return the L2 distance between two vectors."""
    return (a - b).norm()

def right_kernel_two_by_two(A):
    """
    For a 2x2 matrix A over an approximate field like RR or CC, find an
    element in the right kernel.
    """
    prec = A.base_ring().precision()
    epsilon = (2.0)**(-0.8*prec)
    assert A.determinant().abs() < epsilon, 'Matrix looks non-singular'
    a, b = max(A.rows(), key=lambda v: v.norm())
    v = vector([1, -a/b]) if b.abs() > a.abs() else vector([-b/a, 1])
    assert (A*v).norm() < epsilon, 'Supposed kernel vector is not in the kernel.'
    return (1/v.norm())*v

def eigenvectors(A):
    """
    Returns the two eigenvectors of a loxodromic matrix A.
    """
    CC = A.base_ring()
    return [right_kernel_two_by_two(A-eigval) for eigval in A.charpoly().roots(CC, False)]

#def eigenbasis(A, B):
#    """
#    Given loxodromic matrices A and B, return a basis of C^2 consisting of
#   one eigenvector from each.
#    """
#    basis = [ (a, b) for a in eigenvectors(A) for b in eigenvectors(B) ]
#    return matrix(min(basis, key=lambda (a,b) : abs(a*b))).transpose()

def eigenvector(A):
    """
    Returns the eigenvector corresponding to the larger eigenvalue of a
    loxodromic matrix A
    """
    evalues = eigenvalues(A)
    evalue = max(evalues, key=lambda x: x.abs())
    return right_kernel_two_by_two(A - evalue*Id2)

def eigenbasis(A, B):
    """
    Given loxodromic matrices A and B, return a basis of C^2 consisting of
    one eigenvector from each.
    """
    eA = eigenvector(A)
    eB = eigenvector(B)
    return matrix([[eA[0], eB[0]], [eA[1], eB[1]]])
    #return matrix([eigenvector(A), eigenvector(B)]).transpose()

def conjugator_into_PSL2R(A, B):
    """
    Given loxodromic matrices A and B which lie in a common conjugate of
    PSL(2, R), return a matrix C so that C^(-1)*A*C and C^(-1)*B*C are in
    PSL(2, R) itself.
    """
    C = eigenbasis(A, B)
    AA = GL2C_inverse(C)*A*C
    BB = GL2C_inverse(C)*B*C
    a = AA[0, 1]
    b = BB[1, 0]
    if abs(a) > abs(b):
        e, f = 1, abs(a)/a
    else:
        e, f = abs(b)/b, 1

    return C * matrix(A.base_ring(), [[e, 0], [0, f]])


def conjugate_into_PSL2R(rho, max_error, depth=7):
    """
    Given a holonomy representation with generators near PSL(2,R),
    return a list of generators for a conjugate representation with
    image in PSL(2,R).
    """
    gens = tuple(rho.generators())
    new_mats, error = real_part_of_matrices_with_error(rho(g) for g in gens)
    if error < max_error:
        return new_mats

    # Search for two non-commuting conjugate loxodromics
    for word in conjugacy_classes_in_Fn(gens, depth):
        U = rho(word)
        if abs(U.trace()) > 2.0001:
            conjugates = [rho(g)*U*rho(g.upper()) for g in gens]
            V = max(conjugates, key=lambda M, N=U: (N - M).norm())
            comm = U*V*SL2C_inverse(U)*SL2C_inverse(V)
            if abs(comm.trace() - 2) > 1e-10:
                C = conjugator_into_PSL2R(U, V)
                new_mats = [GL2C_inverse(C) * rho(g) * C for g in gens]
                final_mats, error = real_part_of_matrices_with_error(new_mats)
                assert error < max_error, 'Matrices do not seem to be real.'
                return final_mats
    raise CouldNotConjugateIntoPSL2R

def fixed_point(A):
    """
    Return a complex number fixed by this matrix.  In the case of a
    parabolic, the fixed point will be real.
    """
    assert A.trace().abs() <= 2.0, 'Please make sure you have not changed the generators!'
    CC = complex_field(A.base_ring())
    x = pari('x')
    a, b, c, d = [pari(z) for z in A.list()]
    p = c*x*x + (d - a)*x - b
    if p == 0:
        return complex_I(CC)
    fp = max(p.polroots(precision=CC.precision()), key=lambda z: z.imag())
    return CC(fp)

def elliptic_rotation_angle(A):
    """Return the rotation angle of this element at its fixed point."""
    z = fixed_point(A)
    c, d = A.list()[2:]
    derivative = 1/(c*z + d)**2
    pi = get_pi(A.base_ring())
    r = -arg(derivative)
    if r < 0:
        r = r + 2*pi
    return r/(2*pi)

def translation_amount(A_til):
    """Return the translation component of an element of ~PSL(2,R)."""
    return elliptic_rotation_angle(A_til.A) + A_til.s

def rot(R, t, s):
    """Return an element of ~PSL(2,R) lifting a rotation in PSL(2,R)."""
    t = get_pi(R)*R(t)
    A = matrix(R, [[t.cos(), -t.sin()], [t.sin(), t.cos()]])
    return PSL2RtildeElement(A, s)

def shift_of_central(A_til):
    """Verify that this element is in the center and return its shift."""
    assert A_til.is_central(), "Central element isn't really central."
    return A_til.s

def normalizer_wrt_target_meridian_holonomy(meridian_matrix, target):
    """
    The subgroup PSL(2,R) < PSL(2,C) has index 2 in its normalizer
    with the non-trivial coset being represented by the diagonal
    matrix Δ with i and -i on the diagonal.  Conjugation by Δ maps the
    invariant hyperbolic plane to itself by an orientation reversing
    involution, which has the effect of changing the rotation number
    of an elliptic element of PSL(2,R) to its negative.  The matrix
    computed by *conjugate_into_PSL2R* may or may not conjugate each
    element into the trivial coset.

    In order for the translation numbers to have the correct sign, we
    need to adjust the conjugator produced by *conjugate_into_PSL2R*.
    This function implements an imperfect heuristic method of doing
    this.  (See m276 for an example where it fails.)  It returns a
    conjugator to be applied after the one provided by
    conjugate_into_PSL2R.

    As a first approximation, the returned conjugator matrix is either
    Id or Δ, the latter being chosen when the rotation angle of the
    meridian reduced mod 2π lies in the interval (π, 2*π).  (The
    ambiguity for reduced angle π is responsible for the artifact in
    m276.  The ambiguity for angle 2π causes artifacts when parabolics
    are included).

    For a completely independent reason, the actual return value is
    either Id or Δ times the matrix of the inversion z -> -1/z.  The
    purpose of this is to avoid failures in the function *fixed_point*
    which arise when it is passed a parabolic matrix with fixed point
    at infinity, i.e an upper triangular matrix.

    NOTE: When the trace of the meridian is 0 equality holds in the
    comparison used to decide whether to flip; the test is ambiguous
    in this case.  This code arbitrarily chooses never to flip if the
    trace is 0.  That choice can lead to discontinuous translations
    arcs, and must be corrected when computing the translations.
    """
    current = elliptic_rotation_angle(meridian_matrix)
    RR = current.parent()
    CC = complex_field(RR)
    target_arg = CC(target).arg()/(2*get_pi(RR))
    target_arg -= target_arg.floor()
    other = 1 - current
    if abs(other - target_arg) < abs(current - target_arg):
        I = complex_I(CC)
        C = matrix(CC, [[0, I], [I, 0]])
    else:
        C = matrix(CC, [[0, -1], [1, 0]])
    return C

class PSL2RRepOf3ManifoldGroup(PSL2CRepOf3ManifoldGroup):
    """
    >>> import snappy
    >>> M = snappy.Manifold('m004(3,2)')
    >>> M.set_peripheral_curves('fillings')
    >>> shapes = [0.48886560625734599, 0.25766090533555303]
    >>> rho = PSL2RRepOf3ManifoldGroup(M, 1.0, shapes, 250)
    >>> rho
    <m004(1,0): [0.48887, 0.25766]>
    >>> rho.representation_lifts()
    True
    """
    def __init__(self, rep_or_manifold,
                 target_meridian_holonomy=None,
                 rough_shapes=None,
                 precision=None,
                 fundamental_group_args=(True, False, True)):
        if isinstance(rep_or_manifold, PSL2CRepOf3ManifoldGroup):
            rep = rep_or_manifold
        else:
            rep = PSL2CRepOf3ManifoldGroup(
                rep_or_manifold,
                target_meridian_holonomy,
                rough_shapes,
                precision,
                fundamental_group_args)
        self.manifold = rep.manifold
        self.target_meridian_holonomy = rep.target_meridian_holonomy
        self.rough_shapes = rep.rough_shapes
        self.precision = rep.precision
        self.fundamental_group_args = rep.fundamental_group_args
        self._cache = {}

    def polished_holonomy(self, precision=None):
        """Construct and return a polished holonomy with values in PSL2(R)."""
        self._update_precision(precision)
        precision = self.precision
        if precision == None:
            raise ValueError("Need to have a nontrivial precision set")
        mangled = "polished_holonomy_%s" % precision
        if not self._cache.has_key(mangled):
            epsilon = 2.0**(-0.8*precision)
            G = polished_group(self.manifold,
                               self.polished_shapes().shapelist,
                               precision,
                               fundamental_group_args=self.fundamental_group_args,
                               lift_to_SL2=False)
            new_mats = conjugate_into_PSL2R(G, epsilon)
            if self.target_meridian_holonomy:
                meridian_word = self.meridian()
                meridian_matrix = apply_representation(meridian_word, new_mats)
                C = normalizer_wrt_target_meridian_holonomy(meridian_matrix,
                                                            self.target_meridian_holonomy)
                new_mats = real_part_of_matrices_with_error(
                    [SL2C_inverse(C)*M*C for M in new_mats])[0]
                self._new_matrices(G, new_mats)
            if not G.check_representation() < epsilon:
                raise CheckRepresentationFailed
            self._cache[mangled] = G
        return self._cache[mangled]

    @staticmethod
    def _new_matrices(G, new_mats):
        for g in G.generators():
            G._hom_dict[g] = apply_representation(g, new_mats)
            G._hom_dict[g.upper()] = apply_representation(g.upper(), new_mats)
        G._id = Id2

    def flip(self):
        """Conjugate this rep by Δ."""
        G = self.polished_holonomy()
        new_mats = [G(g) for g in G.generators()]
        meridian_word = self.meridian()
        meridian_matrix = self(meridian_word)
        CC = meridian_matrix.base_ring()
        I = complex_I(CC)
        D = matrix(CC, [[I, 0], [0, -I]])
        new_mats = real_part_of_matrices_with_error([-D*M*D for M in new_mats])[0]
        self._new_matrices(G, new_mats)
        mangled = "polished_holonomy_%s" % self.precision
        self._cache[mangled] = G

    def thurston_class_of_relation(self, word, init_pt):
        """
        The Thurston Class is twice the Euler class.  Not sure WTF this means when there's
        2-torsion in H^2.
        """
        n = len(word)
        b = normalize_vector(init_pt)
        points = [apply_matrix(self(word[:i]), b) for i in range(1, n)]
        error = min([vector_dist(b, p) for p in points] +
                    [vector_dist(points[i], points[i + 1]) for i in range(n - 2)])
        return sum([orientation(b, points[i], points[i+1]) for i in range(n - 2)]), error

    def thurston_class(self, init_pt=(2, -3)):
        """Return the Thurston class of this rep."""
        init_pt = vector(self.matrix_field(), init_pt)
        ans = [self.thurston_class_of_relation(R, init_pt) for R in self.relators()]
        thurston, error = [x[0] for x in ans], min([x[1] for x in ans])
        return self.class_in_H2(thurston), error

    def euler_cocycle_on_relations(self):
        """
        Evaluate the euler cocycle on each relation and return the list of values.

        Raise an assertion error if the lift of a relation is not central in the
        lifted free group rep associated to this rep.
        """
        rho_til = LiftedFreeGroupRep(self)
        lifts = [rho_til(R) for R in self.relators()]
        assert False not in [R_til.is_central() for R_til in lifts]
        # Not sure where the sign comes from, but hey.
        return [-R_til.s for R_til in lifts]

    def euler_class(self, double=False):
        """Return the Euler class of this rep."""
        e = self.euler_cocycle_on_relations()
        if double:
            e = [2*x for x in e]
        return self.class_in_H2(e)

    def representation_lifts(self, precision=None):
        """Does this rep lift to ~PSL(2,R)?"""
        self._update_precision(precision)
        thurston, _ = self.thurston_class()
        if False in [x == 0 for x in thurston]:
            return False
        else:
            if not self.has_2_torsion_in_H2():
                return True
            else:
                return False in [x == 0 for x in self.euler_class()]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
