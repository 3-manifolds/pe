# -*- coding: utf-8 -*-
"""
Define the class PSL2RRepOf3ManifoldGroup which represents an
arbitrary precision holonomy representation with image in SL(2,R).
"""

from .sage_helper import (get_pi, matrix, vector, RealField, Id2, complex_I)

from .complex_reps import (PSL2CRepOf3ManifoldGroup, polished_group,
                           apply_representation, inverse_word,
                           GL2C_inverse, SL2C_inverse,
                           CheckRepresentationFailed, words_in_Fn)
from .euler import orientation, PSL2RtildeElement, LiftedFreeGroupRep
from .matrix_helper import (eigenvectors, apply_matrix, vector_dist,
                            normalize_vector, fixed_point)
from . import quadratic_form


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
    assert abs(A[1, 0]) < max_error and abs(abs(A[0, 0]) - 1) < max_error
    assert abs(B[0, 1]) < max_error and abs(abs(B[0, 0]) - 1) < max_error

    # If A and B commute the we're in a degenerate situation and so use
    # a generic algorithm.
    
    if abs((A*B*SL2C_inverse(A)*SL2C_inverse(B)).trace() - 2) < max_error:
        C = quadratic_form.conjugator_into_SL2R(gen_mats)
        Cinv = SL2C_inverse(C)
        curr_mats = [Cinv*M*C for M in gen_mats]
        curr_mats, error = real_part_of_matrices_with_error(curr_mats)
        if error > max_error:
            raise CouldNotConjugateIntoPSL2R
        return curr_mats
    
    # First conjugate so that A is diagonal.
    CC = A.base_ring()
    a0, a1 = A[0]
    z = a0*a1/(1 - a0**2)   # Other fixed point of A
    C = matrix(CC, [[1, z], [0, 1]])
    Cinv = SL2C_inverse(C)
    curr_mats = [Cinv*M*C for M in [A, B] + gen_mats]
    A, B = curr_mats[:2]
    assert abs(A[1, 0]) < max_error and abs(A[0, 1]) < max_error

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
    # use sends (-1, 0, 1, infinity) -> (i, 1, -i, -1).

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
    assert u.imag() > 0 and v.imag() > 0
    assert abs(u*v + 1) < max_error
    return curr_mats[2:]

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
    M.dehn_fill((0, 0))
    M = M.with_hyperbolic_structure()
    assert M.cusp_info('complete?') == [True]
    G = polished_group(M, M.tetrahedra_shapes('rect'), precision=53)
    m_inf, m_0 = None, None
    m = G.peripheral_curves()[0][0]
    for n in range(7):
        if n == 0:
            words = ['']
        else:
            words = [w for w in words_in_Fn(''.join(G.generators()), n) if len(w) == n]
        for w in words:
            m_w = w + m + inverse_word(w)
            A = G.SL2C(m_w)
            if m_inf is None and abs(A[1, 0]) < 1e-6:
                m_inf = m_w
            if m_0 is None and abs(A[0, 1]) < 1e-6:
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
    >>> shapes = [(0.56872407246562728+0.20375919309358881j),\
 (1.8955789278288073-0.89557892782880721j), 0.62339350249879155]
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
                 special_meridians=None,
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
                    A[0, 1], A[1, 0] = -A[0, 1], -A[1, 0]
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
        if any(x != 0 for x in thurston):
            return False
        else:
            if not self.has_2_torsion_in_H2():
                return True
            else:
                return all(x == 0 for x in self.euler_class())

    def lift_on_cusped_manifold(self):
        rel_cutoff = len(self.generators()) - 1
        euler_cocycle = self.euler_cocycle_on_relations()
        D = self.coboundary_1_matrix()[:rel_cutoff]
        M = matrix([euler_cocycle] + D.columns())
        k = M.left_kernel().basis()[0]
        if k[0] != 1:
            # Two reasons we could be here: the euler class isn't zero or
            # the implicit assumption about how left_kernel works is violated.
            # Only the latter is actually worrysome.
            if D.elementary_divisors() == M.transpose().elementary_divisors():
                raise AssertionError('Need better implementation, Nathan')
            else:
                return None
        shifts = (-k)[1:]
        good_lifts = [PSL2RtildeElement(self(g), s)
                      for g, s in zip(self.generators(), shifts)]
        return LiftedFreeGroupRep(self, good_lifts)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
