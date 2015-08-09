"""
Suppose M is the exterior of a knot in S^3 where the trace field
has a real place, and hence a rep into X_R^P, but whose Alexander
polynomial has no unimodular roots.  The first few examples are:

K7_92 7 1 = v2508
K7_117 6 2 = v3195
K7_125 7 1 = v3423

Super slow on computing the trace field:

K7_31 (v0741)
K8_101 (t05252)

"""

import snappy
from sage.all import CC

def unimodular_roots(poly):
    """Roots on the unit circle, repeated with appropriate multiplicity"""
    ans = []
    for z, m in poly.roots(CC):
        if abs(abs(z) - 1) < 1e-10:
            ans += m*[z]
    return ans

def num_unimodular_roots(poly):
    """Count roots on the unit circle, with multiplicity"""
    return len(unimodular_roots(poly))

def unimodular_free_alex_poly():
    return [M.name() for M in snappy.CensusKnots
            if num_unimodular_roots(M.alexander_polynomial()) == 0]

candidates = ['K2_1', 'K4_1', 'K5_2', 'K5_21', 'K6_1', 'K6_34', 'K6_42', 'K6_43', 'K7_1', 'K7_84', 'K7_85', 'K7_89', 'K7_92', 'K7_104', 'K7_117', 'K7_125', 'K7_128', 'K8_1', 'K8_70', 'K8_101', 'K8_116', 'K8_137', 'K8_141', 'K8_151', 'K8_163', 'K8_173', 'K8_234', 'K8_240', 'K8_249', 'K8_252', 'K8_256', 'K8_269', 'K8_277', 'K8_280', 'K8_281', 'K8_291', 'K8_292', 'K8_293', 'K8_294', 'K8_297']

def find_trace_field(M, max_prec=1e4,  optimize_field=False):
    """
    Starts with 100 bits of precision and degree 10 and then doubles
    both successively until it succeeds or max_prec is bit.  The ratio
    of bits/degree is roughly the one recommended in [CGHN]
    """
    traces = M.trace_field_gens()
    prec, deg = 100, 10
    ans = None
    while ans is None and prec <= max_prec:
        ans = traces.find_field(prec, deg, optimize_field, verbosity=False)
        prec, deg = 2*prec, 2*deg
    if ans is None:
        raise ValueError('Could not compute trace field')
    return ans
            
        
def test():
    #for M in snappy.CensusKnots:
    for name in candidates:
        M = snappy.Manifold(name)
        K, places, traces = find_trace_field(M)
        print M.name(), K.degree(), len(K.real_embeddings())


def snap_test():
    for M in snappy.CensusKnots:
        K, places, traces = find_trace_field(M)
        print M.name(), K.degree(), len(K.real_embeddings())
