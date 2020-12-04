# -*- coding: utf-8 -*-

"""
For a representation G -> PSL(2, R) compute the Euler class of the
action on P^1(R).

Initially, tried to follows pages 363-367 of

http://www.umpa.ens-lyon.fr/~ghys/articles/groups-acting-circle.pdf

However, there seems to be an error on page 365 in defining the
cocycle cbar.  In particular, according to Brown's Cohomology
of Groups, the cocycle associated to the an exension should
be defined in terms of the bar notation as

[g1 | g2] = [1, g1, g1 g2]

That is cbar([g1|g2]) = s(g1 g2)^-1 s(g1) s(g2) rather
that the RHS being cbar(g1, g2)
"""
from __future__ import print_function
from .sage_helper import Id2, sqrt
from .matrix_helper import elliptic_rotation_angle, eigenvectors

def sign(a):
    return bool(a > 0) - bool(a < 0)

def wedge(a, b):
    """Return the wedge product of two 2-vectors."""
    return -a[0]*b[1] + a[1]*b[0]

def orientation(a, b, c):
    """Is the oriented triangle Δ(a,b,c) counterclockwise?"""
    return sign(wedge(a, b) * wedge(b, c) * wedge(c, a))

class PointInP1R(object):
    """
    A point in P^1(R), modeled as a point on S^1 in R^2 whose polar
    angle satisfies 0 <= theta < pi.  We view R as the universal cover
    of P^1(R) with the covering map that sends t to the point [x : y]
    where x = 1-2*floor(t) and y=sqrt(1-x^2).

    Instantiate a PointInP1R either by providing a vector v in R^2 - 0
    or a real number t.  In the latter case, the PointInP1R will be
    the image of t under our universal covering map.
    """
    def __init__(self, v=None, t=None):
        if t != None:
            if t == 0:
                R = t.parent()
                self.v = (R(1), R(0))
            else:
                t = t - t.floor()
                x = 1 - 2*t
                y = sqrt(1 - x**2)
                self.v = (x, y)
        else:
            self.v = self._normalize(v)

    @staticmethod
    def _normalize(v):
        a, b = v
        norm = sqrt(a**2 + b**2)
        a, b = a/norm, b/norm
        if b < 0 or (a == -1 and b == 0):
            a, b = -a, -b
        return (a, b)

    def lift(self):
        """The "standard" lift of this point, which lies in [0,1)"""
        return (1-self.v[0])/2

    def __repr__(self):
        return "<[%.5f:%.5f] in P^1(R)>" % self.v

    def __rmul__(self, mat):
        a, b = self.v
        return PointInP1R((mat[0][0]*a + mat[0][1]*b, mat[1][0]*a + mat[1][1]*b))

    def __getitem__(self, i):
        return self.v[i]

def sigma_action(A, x):
    """
    The projective tranformation given by the matrix A has a unique
    "standard" lift sigma_A in Homeo(R), determined by the property
    that sigma_A(0) lies in [0, 1).

    Return sigma_A(x)
    """
    R = x.parent()
    p0, p1 = A*PointInP1R((R(1), R(0))), A*PointInP1R(t=x)
    a1 = p1.lift()
    b1 = a1 if p0[0] >= p1[0] else a1 + 1
    return x.floor() + b1

def eval_cocycle(A, B, AB, x):
    """
    Evaluate the (theoretically constant) cocycle function at x.
    """
    value = sigma_action(A, sigma_action(B, x)) - sigma_action(AB, x)
    rounded = value.round()
    return value, rounded

def univ_euler_cocycle(A, B, samples=3):
    """
    Evaluate the universal euler cocycle on [A | B], the class of (1,
    A, A*B).

    The universal euler cocycle represents the class in H^2(PSL(2,R))
    which corresponds to the central extension
    Z -> ~PSL(2,R) -> PSL(2,R) .

    Here ~PSL(2,R) denotes the subgroup of Homeo+(R) consisting of all lifts
    of elements of PSL(2,R), viewed as a subgroup of Homeo+(S^1)
    """
    R = A.base_ring()
    AB = A*B
    # Not doing this produces lots of artifacts.
    if is_almost_identity(A) or is_almost_identity(B):
        return 0
    if is_almost_identity(AB):
        AB = Id2
    epsilon = R(2.0)**(-R.prec()//2)
    value, rounded = eval_cocycle(A, B, AB, R(0.5))
    if abs(value - rounded) < epsilon:
        return rounded
    # Uh-oh. Apparently we have run into some sort of numerical issue.
    # We'll try sampling our "constant" function at several random points.
    # And let's print something here to see if this ever happens:
    print('Trying random samples!')
    data = set()
    for _ in range(samples):
        x = R.random_element()
        ans, rounded = eval_cocycle(A, B, AB, x)
        if abs(ans - rounded) < epsilon:
            data.add(rounded)
    assert len(data) == 1
    for ans in data:
        return ans

def my_matrix_norm(A):
    """
    Sage converts entries to doubles, which can lead to a huge loss of
    precision here.
    """
    return max(abs(e) for e in A.list())

def is_almost_identity(A, tol=0.8):
    """Decide if a 2x2 matrix is the identity to within round-off error."""
    # First a quick check to rule out most inputs.
    if abs(A[0][1]) > 1e-10 or abs(A[1][0]) > 1e-10:
        return False
    RR = A.base_ring()
    error = min(my_matrix_norm(A - Id2), my_matrix_norm(A + Id2))
    epsilon = RR(2.0)**RR(-tol*RR.prec()).floor()
    return error <= epsilon

class PSL2RtildeElement(object):
    """
    An element of the central extension of SL(2,R) with center Z which
    is determined by the universal euler cocycle.
    """
    def __init__(self, A, s):
        self.A, self.s = A, s

    def __call__(self, x):
        return sigma_action(self.A, x) + self.s

    def inverse(self):
        """The inverse of this element of ~PSL2R,"""
        A = self.A
        Ainv = A.adjugate()
        return PSL2RtildeElement(Ainv, -self.s - univ_euler_cocycle(A, Ainv))

    def __mul__(self, other):
        A, s, B, t = self.A, self.s, other.A, other.s
        return PSL2RtildeElement(A*B, s + t + univ_euler_cocycle(A, B))

    def __repr__(self):
        A_entries = tuple(self.A.list())
        return "<PSL2tilde: A = [[%.5f,%.5f],[%.5f,%.5f]]; s = %s>" % (A_entries + (self.s,))

    def base_ring(self):
        """Return the ring to which entries and shift belong."""
        return self.A.base_ring()

    def is_central(self):
        """Is this element in the center of ~PSL2R?"""
        return is_almost_identity(self.A)

    def translation_number(self):
        """
        The translation number of an element g of ~PSL2R is the limit of
        |g^n(x) - x|/n as n -> infinity, for any x in R.  Note that
        this lifts the usual rotation number on the circle.
        """
        A, s = self.A, self.s
        eigvects = eigenvectors(A)
        if len(eigvects) == 0:
            assert abs(A.trace()) < 2
            return elliptic_rotation_angle(A) + s
        else:
            x = PointInP1R(v=eigvects[0]).lift()
            return self(x) - x

class LiftedFreeGroupRep(object):
    """
    A representation of a free group into ~PSL(2,R).
    """
    def __init__(self, group, images=None):
        self.group = group
        gens = group.generators()
        if images is None:
            images = [PSL2RtildeElement(group(g), 0) for g in gens]
        gen_images = dict()
        for g, Atil in zip(gens, images):
            gen_images[g] = Atil
            gen_images[g.upper()] = Atil.inverse()
        self.gen_images = gen_images

    def __call__(self, word):
        ims = self.gen_images
        ans = ims[word[0]]
        for w in word[1: ]:
            ans = ans * ims[w]
        return ans

    def peripheral_translations(self):
        meridian, longitude = self.group.peripheral_curves()[0]
        trans_M = self(meridian).translation_number()
        trans_L = self(longitude).translation_number()
        return trans_M, trans_L

# These are not currently used.
def eval_thurston_cocycle(A, B, p):
    """Evaluate the Thurston cocycle."""
    return orientation(p, A*p, (A*B)*p)

def thurston_cocycle_of_relation(rho, rel):
    """
    See Jekel, Solomon M. (1989). A simplicial formula and bound for
    the Euler class. Israel J. Math. 66(1-3), 247–259.
    """
    assert len(rel) > 2
    ans, g = [], rho(rel[0])
    R = rho('').base_ring()
    p = PointInP1R(t=R(0))
    for w in rel[1:-1]:
        h = rho(w)
        ans.append(eval_thurston_cocycle(g, h, p))
        g = g*h
    return sum(ans)

