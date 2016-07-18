"""
Computing the signature function of a knot in the 3-sphere.
"""

from sage.all import (ComplexField, RealField, RationalField, LaurentPolynomialRing, arccos)
import snappy

def compact_form(p):
    """
    Given a palindromic polynomial p(x) of degree 2n, return a
    polynomial g so that p(x) = x^n g(x + 1/x).
    """
    coeffs = p.coefficients(sparse=False)
    assert len(coeffs) % 2 == 1
    assert coeffs == list(reversed(coeffs))
    R = p.parent()
    x = R.gen()
    f, g = p, R(0)
    
    while f != 0:
        c = f.leading_coefficient()
        assert f.degree() % 2 == 0
        d = f.degree()//2
        g += c*x**d
        f = (f - c*(x**2 + 1)**d)
        if f != 0:
            e = min(f.exponents())
            assert e > 0 and f % x**e == 0
            f = f // x**e

    # Double check
    L = LaurentPolynomialRing(R.base_ring(), repr(x))
    y = L.gen()
    assert p == y**(p.degree() // 2) * g(y + 1/y)
    return g

def arg_to_circle(angle_mod_one):
    RR = angle_mod_one.parent()
    CC = RR.complex_field()
    pi, i = RR.pi(), CC.gen()
    return (2*pi*i).exp()

def real_root_arguments(p, prec=212):
    """
    Note: if p(x) = g(x + 1/x) then p'(1) = p'(-1) = 0 by the chain
    rule; in particular, 1 and -1 are *never* simple roots of p(x).
    """
    assert p(1) != 0 and p(-1) != 0
    g = compact_form(p)
    RR = RealField(prec)
    roots = sorted([(arccos(x/2), e) for x, e in g.roots(RR) if abs(x) <= 2])
    return roots

M = snappy.Manifold('K14n1234')
L = M.link()

    
    
    
    
