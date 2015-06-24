from sage.all import *
from .polish_reps import (PSL2CRepOf3ManifoldGroup, polished_holonomy,
                         apply_representation, GL2C_inverse, SL2C_inverse,
                         CheckRepresentationFailed, conjugacy_classes_in_Fn)
from . import euler

class CouldNotConjugateIntoPSL2R(Exception):
    pass

def real_part_of_matrix_with_error(A):
    RR = RealField(A.base_ring().precision())
    entries = A.list()
    real_parts, error = [clean_real(x.real()) for x in entries], max([abs(x.imag()) for x in entries])
    B = matrix(RR, real_parts, nrows=A.nrows(), ncols=A.ncols())
    if B.trace() < 0:
        B = -B
    return B, error

def real_part_of_matricies_with_error(matrices):
    real_with_errors = [real_part_of_matrix_with_error(A) for A in matrices]
    return [r for r,e in real_with_errors], max(e for r, e in real_with_errors)

def normalize_vector(v):
    return v/v.norm()

def apply_matrix(mat, v):
    return normalize_vector(mat*v)

def swapped_dot(a, b):
    return -a[0]*b[1] + a[1]*b[0]

def orientation(a, b, c):
    return cmp( swapped_dot(a,b) * swapped_dot(b,c) * swapped_dot(c, a), 0)

def dist(a,b):
    return (a - b).norm()

def clean_real(r):
    RR = r.parent()
    epsilon = RR(2)**(-0.5*RR.precision())
    return RR(0) if abs(r) < epsilon else r
    
def right_kernel_two_by_two(A):
    """
    For a 2x2 matrix A over an approximate field like RR or CC find an
    element in the right kernel.
    """
    CC = A.base_ring()
    prec = CC.precision()
    epsilon = (RealField()(2.0))**(-0.8*prec)
    assert abs(A.determinant()) < epsilon
    a, b = max(A.rows(), key=lambda v:v.norm())
    v = vector(CC, [1, -a/b]) if abs(b) > abs(a) else vector(CC, [-b/a, 1])
    assert (A*v).norm() < epsilon
    return v/v.norm()
    
def eigenvectors(A):
    """
    Returns the two eigenvectors of a loxodromic matrix A
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
    CC = A.base_ring()
    evalues =  A.charpoly().roots(CC, False)
    evalue = max(evalues, key=abs)
    return right_kernel_two_by_two(A - evalue)
    
def eigenbasis(A, B):
    """
    Given loxodromic matrices A and B, return a basis of C^2 consisting of
    one eigenvector from each. 
    """
    return matrix([eigenvector(A), eigenvector(B)]).transpose()


def conjugator_into_PSL2R(A, B):
    """
    Given loxodromic matrices A and B which lie in a common conjugate of
    PSL(2, R), return a matrix C so that C^(-1)*A*C and C^(-1)*B*C are in
    PSL(2, R) itself.
    """
    C = eigenbasis(A, B)
    AA = GL2C_inverse(C)*A*C
    BB = GL2C_inverse(C)*B*C
    a = AA[0,1]
    b = BB[1,0]
    if abs(a) > abs(b):
        e, f = 1, abs(a)/a
    else:
        e, f = abs(b)/b, 1

    return C * matrix(A.base_ring(), [[e, 0], [0, f]])

def conjugate_into_PSL2R(rho, max_error, depth=7):
    gens = rho.generators()
    new_mats, error = real_part_of_matricies_with_error(rho(g) for g in gens)
    if error < max_error:
        return new_mats

    for word in conjugacy_classes_in_Fn(gens, depth):
        U = rho(word)
        if abs(U.trace()) > 2.0001 :
            conjugates = [ rho(g)*U*rho(g.upper()) for g in gens ]
            V = max(conjugates, key=lambda M: (U - M).norm())
            comm = U*V*SL2C_inverse(U)*SL2C_inverse(V)
            if abs(comm.trace() - 2) > 1e-10:
                C =  conjugator_into_PSL2R(U, V)
                new_mats = [GL2C_inverse(C) * rho(g) * C for g in gens]
                final_mats, error = real_part_of_matricies_with_error(new_mats)
                assert error < max_error
                return final_mats

    raise CouldNotConjugateIntoPSL2R


def elliptic_fixed_point(A):
    assert abs(A.trace()) < 2.0
    RR = A.base_ring()
    CC = RR.complex_field()
    x = PolynomialRing(RR, 'x').gen()
    a, b, c, d = A.list()
    p = c*x*x + (d - a)*x - b
    if p == 0:
        return CC.gen()
    return max(p.roots(CC, False), key=lambda z:z.imag())

def elliptic_rotation_angle(A):
    z = elliptic_fixed_point(A)
    
    a, b, c, d = A.list()
    derivative = 1/(c*z + d)**2
    pi = A.base_ring().pi()
    r = -derivative.argument()
    if r < 0:
        r = r + 2*pi
    return r/(2*pi)

def translation_amount(A_til):
    return elliptic_rotation_angle(A_til.A) + A_til.s

def rot(R, t, s):
    t = R.pi()*R(t)
    A = matrix(R, [[cos(t), -sin(t)], [sin(t), cos(t)]])
    return euler.PSL2RtildeElement(A, s)

def shift_of_central(A_til):
    assert A_til.is_central()
    return A_til.s

def normalizer_wrt_target_meridian_holonomy(meridian_matrix, target):
    current = elliptic_rotation_angle(meridian_matrix)
    CC = current.parent()
    target *= 1/(2*CC.pi())
    target += -target.floor()
    other = 1 - current
    if abs(other - target) < abs(current - target):
        I = CC.gen()
        C =  matrix(CC, [[I, 0], [0, -I]])
    else:
        C = matrix(CC, [[1, 0], [0, 1]])
    return C * matrix(CC, [[1, 1], [1, 2]])
        
    
    
class PSL2RRepOf3ManifoldGroup(PSL2CRepOf3ManifoldGroup):
    """
    >>> import snappy
    >>> M = snappy.Manifold('m004(3,2)')
    >>> M.set_peripheral_curves('fillings')
    >>> shapes = [0.48886560625734599, 0.25766090533555303]
    >>> rho = PSL2RRepOf3ManifoldGroup(M, 0, shapes, 250, [False, True, False])
    >>> rho
    <m004(1,0): [0.48887, 0.25766]>
    >>> rho.representation_lifts()
    True
    """
    def __init__(self, rep_or_manifold,
                 target_meridian_holonomy_arg=None,
                 rough_shapes=None,
                 precision=None,
                 fundamental_group_args=tuple()):
        if isinstance(rep_or_manifold, PSL2CRepOf3ManifoldGroup):
            rep = rep_or_manifold
        else:
           rep = PSL2CRepOf3ManifoldGroup(
               rep_or_manifold,
               target_meridian_holonomy_arg,
               rough_shapes,
               precision,
               fundamental_group_args)
        self.manifold = rep.manifold
        self.target_meridian_holonomy_arg = rep.target_meridian_holonomy_arg
        self.rough_shapes =  rep.rough_shapes
        self.precision = rep.precision
        self.fundamental_group_args = rep.fundamental_group_args
        self._cache = {}

    def polished_holonomy(self, precision=None):
        self._update_precision(precision)
        precision = self.precision
        if precision == None:
            raise ValueError, "Need to have a nontrivial precision set"
        mangled = "polished_holonomy_%s" % precision
        if not self._cache.has_key(mangled):
            epsilon = RR(2.0)**(-0.8*precision)
            G = polished_holonomy(self.manifold, self.target_meridian_holonomy_arg,
                                     precision,
                                     fundamental_group_args=self.fundamental_group_args,
                                     lift_to_SL2=False,
                                     ignore_solution_type=True)
            new_mats = conjugate_into_PSL2R(G, epsilon)
            if self.target_meridian_holonomy_arg:
                meridian_word = self.meridian()
                meridian_matrix = apply_representation(meridian_word, new_mats)
                C = normalizer_wrt_target_meridian_holonomy(meridian_matrix,
                                                       self.target_meridian_holonomy_arg)
                new_mats = real_part_of_matricies_with_error([SL2C_inverse(C)*M*C for M in new_mats])[0]
            
            def rho(word):
                return apply_representation(word, new_mats)
            G.SL2C = rho
            if not G.check_representation() < epsilon:
                  raise CheckRepresentationFailed
            self._cache[mangled] = G
                    
        return self._cache[mangled]

    def thurston_class_of_relation(self, word, init_pt):
        """
        The Thurston Class is twice the Euler class.  Not sure WTF this means when there's
        2-torsion in H^2.  
        """
        n = len(word)
        b = normalize_vector(init_pt)
        points = [apply_matrix(self(word[:i]), b) for i in range(1, n)]
        error = min( [dist(b, p) for p in points] + [dist(points[i], points[i + 1]) for i in range(n - 2)])
        return sum( [ orientation(b, points[i], points[i+1]) for i in range(n - 2)]), error

    def thurston_class(self, init_pt = (2,-3)):
        init_pt = vector(self.matrix_field(), init_pt)
        ans = [self.thurston_class_of_relation(R, init_pt) for R in self.relators()]
        thurston, error = vector(ZZ, [x[0] for x in ans]), min([x[1] for x in ans])
        return self.class_in_H2(thurston), error

    def euler_class(self, double=False):
        rels = self.relators()
        e = vector(ZZ, [euler.euler_cocycle_of_relation(self, R) for R in rels])
        if double:
            e = 2*e
        return self.class_in_H2(e)

    def representation_lifts(self, precision=None):
        self._update_precision(precision)
        thurston, error = self.thurston_class()
        if thurston == 0:
            if not self.has_2_torsion_in_H2():
                return True
            else:
                return self.euler_class() == 0

        return False

    def lift_on_cusped_manifold(rho):
        rel_cutoff = len(rho.generators()) - 1
        rels = rho.relators()[:rel_cutoff ]
        euler_cocycle = [euler.euler_cocycle_of_relation(rho, R) for R in rels]
        D = rho.coboundary_1_matrix()[:rel_cutoff]
        M = matrix(ZZ, [euler_cocycle] + D.columns())
        k = M.left_kernel().basis()[0]
        if k[0] != 1:
            # Two reasons we could be here: the euler class isn't zero or
            # the implicit assumption about how left_kernel works is violated.
            # Only the latter is actually worrysome.
            if D.elementary_divisors() == M.transpose().elementary_divisors():
                raise AssertionError('Need better implementation, Nathan')
            else:
                return 
        shifts = (-k)[1:]
        good_lifts = [euler.PSL2RtildeElement(rho(g), s)
                      for g, s in zip(rho.generators(), shifts)]
        rho_til= euler.LiftedFreeGroupRep(rho, good_lifts)
        return rho_til

if __name__ == '__main__':
    import doctest
    doctest.testmod()
