# -*- coding: utf-8 -*-
"""
Define the class PSL2RRepOf3ManifoldGroup which represents an arbitrary precision
holonomy representation with image in SL(2,R).
"""

from .sage_helper import _within_sage, get_pi
from .complex_reps import (PSL2CRepOf3ManifoldGroup, polished_group,
                           apply_representation, inverse_word, GL2C_inverse, SL2C_inverse,
                           CheckRepresentationFailed, words_in_Fn)
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


def conjugate_into_PSL2R(rho, max_error, (m_inf, m_0)):
    # If all shapes are flat, or equivalently if the peripheral holonomy is
    # hyperbolic or parabolic, then there's nothing to do:
    gens = tuple(rho.generators())
    gen_mats = [rho(g) for g in gens]
    new_mats, error = real_part_of_matrices_with_error(gen_mats)
    if error < max_error:
        return new_mats

    # Now for the harder case of elliptic peripheral holonomy
    A, B = rho(m_inf), rho(m_0)
    assert abs(A[1, 0]) < max_error and abs(abs(A[0,0]) - 1) < max_error
    assert abs(B[0, 1]) < max_error and abs(abs(B[0,0]) - 1) < max_error
    A[1,0], B[0, 1] = 0, 0

    # First conjugate so that A is diagonal. 
    CC = A.base_ring()
    a0, a1 = A[0]
    z =  a0*a1/(1 - a0**2)   # Other fixed point of A
    C = matrix(CC, [[1, z], [0, 1]])
    Cinv = SL2C_inverse(C)
    curr_mats = [Cinv*M*C for M in [A, B] + gen_mats]
    A, B = curr_mats[:2]
    assert A[1,0] == 0 and abs(A[0,1]) < max_error
    A[0, 1] = 0

    # The hyperplane P preserved by rho must be orthogonal to the axis
    # of A, which has endpoints 0 and infinity in S^2.  Thus P must
    # correspond to some circle about the origin in the complex
    # plane. The axis of B must also be orthogonal to P, which forces
    # its endpoints to lie on a common ray from the origin in the
    # complex plane.  We next rotate the complex plane so that B's
    # fixed points are on the real axis and are symmetric with respect
    # to inversion in the unit circle.  This will cause rho will
    # preserve the hyperplane over the unit circle about the origin.

    vec0, vec1 = eigenvectors(B)
    pt0, pt1 = vec0[0]/vec0[1], vec1[0]/vec1[1]
    s = abs(pt1/pt0).sqrt()
    u = s*pt0
    C = matrix(CC, [[u, 0], [0, 1]])
    Cinv = matrix(CC, [[1/u, 0], [0, 1]])
    curr_mats = [Cinv*M*C for M in curr_mats]
    A, B = curr_mats[:2]

    # Check we did everything correctly
    vec0, vec1 = eigenvectors(B)
    pt0, pt1 = vec0[0]/vec0[1], vec1[0]/vec1[1]
    assert abs(pt0.imag()) < max_error and abs(pt1.imag()) < max_error
    assert pt0.real() > 0 and pt1.real() > 0
    assert abs(pt0*pt1 - 1) < max_error

    # Now exchange the hyperplane over the unit circle with the one
    # over the real line so that we end up in PSL(2, R).  The map we
    # use sends (-1, 0, 1, infinity) -> (i, 1, -i, -1) 

    i = complex_I(CC)
    C = matrix(CC, [[1, -i], [-1, -i]])
    Cinv = GL2C_inverse(C)
    curr_mats = [Cinv*M*C for M in curr_mats]
    A, B = curr_mats[:2]

    # Check that A fixes i in the upper halfspace model and the
    # ellptic fixed point of B is below it on the imaginary axis.
    
    curr_mats, error = real_part_of_matrices_with_error(curr_mats)
    if error > max_error:
        raise CouldNotConjugateIntoPSL2R
    A, B = curr_mats[:2]
    u, v = fixed_point(A), fixed_point(B)
    assert abs(u.real()) < max_error and abs(v.real()) < max_error
    assert abs(u.imag() - 1) < max_error and v.imag() > 0

    # Do a real dilation so that the fixed points of A and B are on
    # the imaginary axis and equidistant from i.  This is done so that
    # the representations into PSL(2, R) are nearly continous when you
    # have a family with elliptic peripheral holonomy that limits on a
    # representation with parabolic peripheral holonomy.

    C = matrix([[abs(v).sqrt(), 0], [0, 1]])
    Cinv = GL2C_inverse(C)
    curr_mats = [Cinv*M*C for M in curr_mats]
    A, B = curr_mats[:2]
    u, v = fixed_point(A), fixed_point(B)
    assert abs(u.real()) < max_error and abs(v.real()) < max_error
    assert u.imag()  > 0 and v.imag() > 0
    assert abs(u*v + 1) < max_error
    return curr_mats[2:]

def fixed_point(A):
    """
    Return a complex number fixed by the linear fractional
    transformation given by a matrix A.  In the case of a parabolic,
    the fixed point will be real, unless the parabolic fixed point is
    aat infinity in which case all hell breaks loose.
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
    """Return the rotation angle of an element of PSL(2,R)at its fixed point."""
    z = fixed_point(A)
    c, d = A.list()[2:]
    derivative = 1/(c*z + d)**2
    pi = get_pi(A.base_ring())
    r = -arg(derivative)
    if r < 0:
        r = r + 2*pi
    return r/(2*pi)

def translation_of_lifted_rotation(R_til):
    """
    Return the translation amount of this element of ~PSL2R.

    NOTE: This method Assumes that this element lifts a rotation
    matrix in SL2R!
    """
    return elliptic_rotation_angle(R_til.A) + R_til.s

def rot(R, t, s):
    """Return an element of ~PSL(2,R) lifting a rotation in PSL(2,R)."""
    t = get_pi(R)*R(t)
    A = matrix(R, [[t.cos(), -t.sin()], [t.sin(), t.cos()]])
    return PSL2RtildeElement(A, s)

def shift_of_central(A_til):
    """Verify that this element is in the center and return its shift."""
    assert A_til.is_central(), "Central element isn't really central."
    return A_til.s

def meridians_fixing_infinity_and_zero(manifold):
    M = manifold.without_hyperbolic_structure()
    M.dehn_fill((0,0))
    M = M.with_hyperbolic_structure()
    assert M.cusp_info('complete?') == [True]
    G = M.fundamental_group(False, False, False)
    m_inf, m_0 = None, None
    m = G.meridian()
    for n in range(7):
        if n == 0:
            words = ['']
        else:
            words = [w for w in words_in_Fn(''.join(G.generators()), n) if len(w) == n]
        for w in words:
            m_w = w + m + inverse_word(w)
            A = G.SL2C(m_w)
            if m_inf is None and abs(A[1,0]) < 1e-6:
                m_inf = m_w
            if m_0 is None and abs(A[0,1]) < 1e-6:
                m_0 = m_w
            if None not in [m_inf, m_0]:
                return m_inf, m_0
    raise ValueError('Could not find desired meridians')

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

    Now an example which is elliptic on the boundary. 

    >>> N = snappy.Manifold('m016')
    >>> shapes = [(0.56872407246562728+0.20375919309358881j), (1.8955789278288073-0.89557892782880721j), 0.62339350249879155]
    >>> psi = PSL2RRepOf3ManifoldGroup(N, -1j, shapes, 250)
    >>> psi
    <m016(0,0): [0.56872+0.20376I, 1.8956-0.89558I, 0.62339]>
    >>> psi.representation_lifts()
    True
    """
    
    def __init__(self, rep_or_manifold,
                 target_meridian_holonomy=None,
                 rough_shapes=None,
                 precision=None,
                 fundamental_group_args=(False, False, False),
                 special_meridians = None,
                 flip=False):
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
        if special_meridians is None:
            special_meridians = meridians_fixing_infinity_and_zero(self.manifold)
        self.meridians = special_meridians
        self._flip = flip

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
            new_mats = conjugate_into_PSL2R(G, epsilon, self.meridians)
            if self._flip:  # Reverse the orientation of H^2
                for A in new_mats:
                    A[0,1], A[1,0] = -A[0, 1], -A[1,0]
            self._new_matrices(G, new_mats)
            if not G.check_representation() < epsilon:
                raise CheckRepresentationFailed
            self._cache[mangled] = G
        return self._cache[mangled]

    def flip(self):
        """Conjugate this rep by Δ, reversing the orientation on H^2"""
        self._flip = not self._flip
        self._cache = {}
        
    @staticmethod
    def _new_matrices(G, new_mats):
        for g in G.generators():
            G._hom_dict[g] = apply_representation(g, new_mats)
            G._hom_dict[g.upper()] = apply_representation(g.upper(), new_mats)
        G._id = Id2

    def thurston_class_of_relation(self, word, init_pt):
        """
        The Thurston Class is twice the Euler class.  Not sure WTF this
        means when there's 2-torsion in H^2.
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
