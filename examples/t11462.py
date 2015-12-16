import os, sys, re, snappy
import phc
from sage.all import CC, RR, QQ, PolynomialRing, matrix, prod

def inverse_sl2(x):
    x0, x1, x2, x3 = x.list()
    return matrix([[x3, -x1], [-x2, x0]])

M = snappy.Manifold('t11462')
G = M.fundamental_group()
periph = G.peripheral_curves()[0]

R = PolynomialRing(QQ, ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3',
                        'c0', 'c1', 'c2', 'c3'])

a = matrix(R, 2, 2, R.gens()[:4])
b = matrix(R, 2, 2, R.gens()[4:8])
c = matrix(R, 2, 2, R.gens()[8:12])
A = inverse_sl2(a)
B = inverse_sl2(b)
C = inverse_sl2(c)

def eval_word(word):
    mats = {'a':a, 'b':b, 'c':c, 'A':A, 'B':B, 'C':C}
    return prod([mats[x] for x in word])

I = R.ideal([a.det() - 1, b.det() - 1, c.det() - 1] + 
            sum([(eval_word(R) - 1).list() for R in G.relators()], []) + 
            [eval_word(w)[1, 0] for w in periph] +
            [eval_word(w)[0, 0]  - 1 for w in periph] +
            [eval_word(w)[1, 1]  - 1 for w in periph] +
            [eval_word(periph[0])[0,1] - 1, eval_word('a')[0,1]])

# Magma confirms that I has dimension 0 via:
#
# sage: magma(I).Dimension()
# 0
#
# So there is a parbolic representation to SL(2, C) where every
# parabolic has trace +2, and so ptolemy really is missing a rep.  I
# didn't check that the rep lands in SL(2, R).

# Doesn't work because of PHC error "raised STORAGE_ERROR : stack
# overflow" when creating the polynomials.
def convert_to_PHC(ideal):
    ideal = I
    R = ideal.ring()
    S = phc.PolyRing(R.variable_names())
    polys = [phc.PHCPoly(S, repr(p)) for p in ideal.gens()]
    system = phc.PHCSystem(S, polys)
