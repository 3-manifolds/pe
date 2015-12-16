"""Exploring the character varieties and such for m389."""

import os, sys, re, snappy
from snappy.snap.polished_reps import ManifoldGroup
from sage.all import CC, RR, QQ, PolynomialRing, matrix


"""
Computing the character variety of a 2-generator group
"""

def reduce_word(word):
    """Cancels inverse generators."""
    ans, progress = word, 1
    while progress:
        ans, progress =  re.subn("aA|Aa|bB|Bb", "", ans)
    return ans

def inverse_word(word):
    L = list(word.swapcase())
    L.reverse()
    return "".join(L)

def some_letter_repeats(word):
    repeats = [L for L in ['a', 'b', 'A', 'B'] if word.count(L) > 1]
    return repeats[0] if len(repeats) > 0 else None
    
def tr(word):
    R = PolynomialRing(QQ, ['x','y','z'])
    x, y, z = R.gens()
    simple_cases = {'':2, 'a':x, 'A':x, 'b':y, 'B':y, 'ab': z, 'ba': z}
    if simple_cases.has_key(word):
        return simple_cases[word]
    L = some_letter_repeats(word)
    if L != None:
        i = word.find(L) 
        w = word[i:] + word[:i]
        j = w[1:].find(L) + 1
        w1, w2 = w[:j], w[j:]
    else:   # Reduce the number of inverse letters
        i = [ i for i, L in enumerate(word) if L == L.upper()][0]
        w = word[i:] + word[:i]
        w1, w2 = w[:1], w[1:]
    return tr(w1) * tr(w2) - tr(reduce_word(inverse_word(w1) + w2))

def components_containing_irreducible_reps(G):
    R = PolynomialRing(QQ, ['x','y','z'])
    assert G.generators() == ['a', 'b'], len(G.relators()) == 1
    r = G.relators()[0]
    I = R.ideal( [tr(r[:j]) - tr(r[j:]) for j in range(len(r))] +
                 [tr('a' + r) - tr('a' + inverse_word(r)), tr('b' + r) - tr('b' + inverse_word(r))]).radical()
    contains_irreducible_rep = [J for J in I.primary_decomposition()
                                if not tr('abAB') - 2 in J]
    return contains_irreducible_rep


"""
Example: m389
"""

M = snappy.Manifold('m389')
G = M.fundamental_group()
assert G.relators() == ['aaabbbaBBAABBabbb']
assert G.peripheral_curves() == [('BBabbba', 'aabbbbbaabb')]

# Rewritten version a_new = A and b_new = ab 

F = PolynomialRing(QQ, 'v').fraction_field()
FC = PolynomialRing(CC, 'v').fraction_field()
v = F.gen()

#MG = ManifoldGroup( ['a', 'b'], ['aaababbbab'], matrices=
#                    [matrix(F, [[v, 1], [0, 1/v]]),
#                     matrix(F, [[1/v, 0], [-(2 +v**2 + 2*v**4)/(1 + v**2 + v**4), v]])])
#assert MG(MG.relators()[0]) == 1
components = components_containing_irreducible_reps(G)
#assert len(components) == 1 and components[0].genus() == 0
#J = components[0].radical()a
#x, y, z = J.ring().gens()

# x = u, y = u, z = 1/(u^2 - 1)
#u = PolynomialRing(QQ, 'u').fraction_field().gen()
#def trace_to_rational(poly):
#    return poly.subs(x=u, y=u, z=(u**2 - 1)**-1)


# There are 6 reducible reps, of which the positive ones have u in:
#
# [0.618033988749895, 1.22474487139159, 1.61803398874989]
#
# The two outer ones have u = z + 1/z where z is a tenth root of
# unity. 

#red_reps = trace_to_rational(tr('abAB') - 2).numerator().roots(CC, False)
#red_reps = [r for r in red_reps if r >= 0]

# Peripheral curves

#m, l = 'aabab', 'aaaba'
#assert MG(m + l) - MG(l + m) == 0
#tr_m = trace_to_rational(tr(m))
#tr_l = trace_to_rational(tr(l))

# Parabolic representations, of which there are 10.  Of the 5 with
# real part >= 0, two are also reducible.  These reducibles include
# representations that are non-trivial on the boundary.  The other
# three parabolic representations are the discrete faithful and its
# Galois conjugates.
#
# [0.618033988749895,
#  1.20556943040059,
#  1.61803398874989,
#  1.10278471520030 - 0.665456951152813*I,
#  1.10278471520030 + 0.665456951152813*I]


#m_para_cond = trace_to_rational(tr_m**2 - 4).numerator()
#l_para_cond = trace_to_rational(tr_l**2 - 4).numerator()
#assert m_para_cond == l_para_cond

#para_reps = [z for z in m_para_cond.roots(CC, False) if z.real() >= 0]

# Which positive real u correspond to SU(2) reps?   

#test_points = [RR(0.005*i) for i in range(0, 400)]
#su2_rep = [(t, su2.conjugate_into_SU2((t, t, (t**2 - 1)**-1))) for t in test_points]

# Which positive real u are peripherally elliptic

#PE = [t for t in test_points if -2 < tr_m(t) < 2 and -2 < tr_l(t) < 2]


