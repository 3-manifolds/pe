"""
For a representation G -> PSL(2, R) compute the
Euler class of the action on P^1(R).

Initially, tried to follows pages 363-367 of

http://www.umpa.ens-lyon.fr/~ghys/articles/groups-acting-circle.pdf

However, there seems to be an error 365 in defining the
cocycle cbar.  In particular, according to Brown's Cohomology
of Groups, the cocycle associated to the an exention should
be defined in terms of the bar notation

[g1 | g2] = [1, g1, g1 g2]

That is cbar([g1|g2]) = s(g1 g2)^-1 s(g1) s(g2) rather
that the RHS being cbar(g1, g2)


"""
# Switching to the below causes crashes elsewhere. Weird.
# from sage.all import matrix, vector, sqrt, arccos, floor, cos, sin
from sage.all import *

def swapped_dot(a, b):
    return -a[0]*b[1] + a[1]*b[0]

def norm(v):
    return sqrt(v[0]*v[0]+ v[1]*v[1])

def orientation(a, b, c):
    return cmp( swapped_dot(a,b) * swapped_dot(b,c) * swapped_dot(c, a), 0)

def clean_real(r):
    RR = r.parent()
    epsilon = RR(2)**(-0.8*RR.precision())
    return RR(0) if abs(r) < epsilon else r

def SL2_inverse(A):
    return matrix([[A[1,1], -A[0,1]], [-A[1,0], A[0, 0]]])

class PointInP1R():
    """
    A point in P^1(R), stored as a point on S^1 in R^2
    where the angle satisfies 0 <= t < pi.  
    """
    def __init__(self, v=None, t=None):
        if t != None:
            R = t.parent()
            theta = (t - t.floor())*R.pi()
            if theta == 0:
                v = vector(R, (1, 0) )
            else:
                v = vector(R, (cos(theta), sin(theta)))

        self.v = self.normalize(vector(v))

    def normalize(self, v):
        v = v/norm(v)
        if v[1] < 0:
            v = -v
        R = v.base_ring()
        if v[0] == R(-1) and v[1] == R(0):
            v = -v 
        return v

    def angle(self):
        return arccos(self.v[0])

    def normalized_angle(self):
        theta = self.angle()
        R = theta.parent()
        return theta/R.pi()
    
    def __repr__(self):
        return "<%.5f in P^1(R)>" % self.normalized_angle()

    def __rmul__(self, mat):
        return PointInP1R(mat * self.v)

    def __getitem__(self, i):
        return self.v[i]

def sigma_action(A, x):
    """
    For the projective tranformation given by the matrix
    A, there is a unique lift sigma(A) to Homeo(R) where
    sigma(A)(0) is in [0, 1)
    """
    R = x.parent()
    p0, p1 = A*PointInP1R( vector(R, (1,0) )), A*PointInP1R(t=x)
    a0, a1 = p0.normalized_angle(), p1.normalized_angle()
    b1 = a1 if a0 <= a1 else a1 + 1
    return x.floor() + b1


def univ_euler_cocycle(f1, f2, samples=3):
    """
    Returns the value of the euler cocycle
    on [f1 | f2] = (1, f1, f1*f2).
    To catch potential numerical issues
    related to cutting S^1 into [0, 1),
    it samples the homeomorphism at several points
    and requires that the results all agree. 
    """

    if samples > 1:
        data = [univ_euler_cocycle(f1, f2, samples=1) for i in range(samples)]
        assert len(set(data)) == 1
        return data[0]

    R = f1.base_ring()
    if is_almost_identity(f1) or is_almost_identity(f2):
        return ZZ(0)
    epsilon = R(2)**(-R.prec()//2) 
    x = R.random_element()
    y = sigma_action(f1, sigma_action(f2, x))
    if is_almost_identity(f1*f2):
        z = x
    else:
        z = sigma_action(f1*f2, x)
    s = y - z
    ans = s.round()
    assert abs(s - ans) < epsilon
    return ans

def my_matrix_norm(A):
    """
    Sage converts entries to CDF which can lead to a
    huge loss of precision here. 
    """
    return max( abs(e) for e in A.list() )
    
def is_almost_identity(A, tol=0.8):
    RR = A.base_ring()
    error = min( my_matrix_norm(A - RR(1)),
                 my_matrix_norm(A+RR(1)))
    epsilon = RR(2)**floor(-tol*RR.prec())
    return error <= epsilon

class PSL2RtildeElement:
    def __init__(self, A, s):
        self.A, self.s = A, s

    def __call__(self, x):
        return sigma_action(self.A, x) + self.s

    def inverse(self):
        A, s = self.A, self.s
        Ainv = SL2_inverse(A)
        return PSL2RtildeElement(Ainv, -self.s - univ_euler_cocycle(A, Ainv))

    def __mul__(self, other):
        A, s, B, t= self.A, self.s, other.A, other.s
        return PSL2RtildeElement(A*B, s + t + univ_euler_cocycle(A,B))

    def __repr__(self):
        A_entries = tuple(self.A.list())
        return "<PSL2tilde: A = [[%.5f,%.5f],[%.5f,%.5f]]; s = %s>" % (A_entries + (self.s,))

    def base_ring(self):
        return self.A.base_ring()

    def is_central(self):
        return is_almost_identity(self.A)


def thurston_cocycle_of_homs(f1, f2, b, samples=3):
    return orientation(b, f1*b, (f1*f2)*b)

class LiftedFreeGroupRep:
    def __init__(self, group, images=None):
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
        
    
def euler_cocycle_of_relation(rho, rel):
    """
    Not sure where the sign comes from, but hey. 
    """
    if isinstance(rho, LiftedFreeGroupRep):
        rho_til = rho
    else:
        rho_til = LiftedFreeGroupRep(rho)
    R_til = rho_til(rel)
    assert R_til.is_central()
    return -R_til.s
    

def thurston_cocycle_of_relation(rho, rel):
    assert len(rel) > 2
    ans, g = [], rho(rel[0])
    R = rho('').base_ring()
    b = PointInP1R(t=R(0))
    for w in rel[1:-1]:
        h = rho(w)
        ans.append(thurston_cocycle_of_homs(g, h, b))
        g = g*h
    return sum(ans)

