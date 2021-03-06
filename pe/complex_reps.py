from __future__ import print_function
import random, string
from itertools import chain
from .shape import ShapeSet, PolishedShapeSet
from .sage_helper import (
    _within_sage, cached_function, RR, CC, Id2, elementary_divisors,
    smith_normal_form, pari, matrix, vector)
from .matrix_helper import SL2C_inverse, GL2C_inverse
from .quadratic_form import preserves_hermitian_form
from snappy.snap.t3mlite import Mcomplex
from snappy.snap.t3mlite.simplex import V0, V1, V2, V3, E01
from snappy.snap.kernel_structures import (
    Infinity, TransferKernelStructuresEngine)
from snappy.snap.fundamental_polyhedron import FundamentalPolyhedronEngine
from snappy.snap.polished_reps import clean_matrix, ManifoldGroup, prod

if _within_sage:
    coboundary_matrix = matrix
else:
    coboundary_matrix = pari.matrix

def random_word(letters, N):
    return ''.join([random.choice(letters) for _ in range(N)])

def inverse_word(word):
    return word.swapcase()[::-1]

@cached_function
def words_in_Fn(gens, n):
    next_letter = dict()
    sym_gens = list(gens) + [g.swapcase() for g in gens]
    for g in sym_gens:
        next_letter[g] = [h for h in sym_gens if h != g.swapcase()]
    if n == 1:
        return sym_gens
    else:
        words = words_in_Fn(gens, n - 1)
        ans = []
        for word in words:
            ans += [word + g for g in next_letter[word[-1]] if len(word) == n - 1]
        return words + ans

def is_lex_first_in_conjugacy_class(word):
    if word[0] == word[-1].swapcase():
        return False
    for i in range(len(word)):
        other = word[i:] + word[:i]
        if other < word or other.swapcase() < word:
            return False
    return True

@cached_function
def conjugacy_classes_in_Fn(gens, n):
    return [word for word in words_in_Fn(gens, n) if is_lex_first_in_conjugacy_class(word)]


class CheckRepresentationFailed(Exception):
    pass

def apply_representation(word, gen_images):
    gens = string.ascii_lowercase[:len(gen_images)]
    rho = dict([(g, gen_images[i]) for i, g in enumerate(gens)] +
               [(g.upper(), SL2C_inverse(gen_images[i])) for i, g in enumerate(gens)])
    return prod([rho[g] for g in word], Id2)

def polished_group(M, shapes, precision=100,
                   fundamental_group_args=(False, False, False),
                   lift_to_SL2=True, check=True):
    error = pari(2.0)**(-precision*0.8)
    G = M.fundamental_group(*fundamental_group_args)
    m = Mcomplex(M)
    f = FundamentalPolyhedronEngine(m)
    t = TransferKernelStructuresEngine(m, M)
    t.add_shapes(shapes)
    t.choose_and_transfer_generators(compute_corners=True,
                                     centroid_at_origin=False)
    T = m.ChooseGenInitialTet
    z = T.ShapeParameters[E01]
    init_tet_vertices = {V0:0, V1:Infinity, V2:z, V3:1}
    f.unglue()
    f.visit_tetrahedra_to_compute_vertices(T, init_tet_vertices)
    f.compute_matrices(normalize_matrices=True)
    mats = f.matrices_for_presentation(G, match_kernel=False)
    clean_mats = [clean_matrix(A, error=error, prec=precision) for A in mats]
    PG = ManifoldGroup(
        G.generators(), G.relators(), G.peripheral_curves(), clean_mats)
    if lift_to_SL2:
        PG.lift_to_SL2C()
    else:
        if check and not PG.is_projective_representation():
            raise CheckRepresentationFailed
    return PG

def format_complex(z, digits=5):
    conv = '%.' + repr(digits) + 'g'
    ten = RR(10.0)
    z = CC(z)
    real = conv % z.real()
    if abs(z.imag()) < ten**-(digits):
        return real
    if abs(z.real()) < ten**-(digits):
        return conv % z.imag() + 'I'
    im = conv % float(abs(z.imag())) + 'I'
    conn = '-' if z.imag() < 0 else '+'
    return real + conn + im

class PSL2CRepOf3ManifoldGroup(object):
    """
    Throughout precision is in bits.

    >>> import snappy
    >>> M = snappy.Manifold('m004')
    >>> rho = PSL2CRepOf3ManifoldGroup(M, 1.0, precision=100)
    >>> rho
    <m004(0,0): [0.5+0.86603I, 0.5+0.86603I]>
    >>> G = rho.polished_holonomy()
    >>> float(G('ab').trace().real()**2)
    4.0
    """
    def __init__(self, manifold,
                 target_meridian_holonomy=None,
                 rough_shapes=None,
                 precision=100,
                 fundamental_group_args=tuple()):
        self.precision = precision
        self.manifold = manifold.copy()
        if rough_shapes is not None:
            self.manifold.set_tetrahedra_shapes(rough_shapes, rough_shapes)
        else:
            rough_shapes = manifold.tetrahedra_shapes('rect')
        self.rough_shapes = ShapeSet(self.manifold, rough_shapes)
        if target_meridian_holonomy is None:
            Hm = manifold.cusp_info('holonomies')[0][0]
            target_meridian_holonomy = (2*pari.pi()*pari('I')*Hm).exp()
        self.target_meridian_holonomy = target_meridian_holonomy
        self.fundamental_group_args = fundamental_group_args
        self._cache = {}

    def __repr__(self):
        return ("<%s" % self.manifold + ": [" +
                ", ".join([format_complex(z) for z in self.rough_shapes]) + "]>")

    def _update_precision(self, precision):
        if precision is not None:
            self.precision = precision

    def advance_holonomy(self, p, q):
        """Change the target_holonomy by exp(2 pi i p/q)"""
        shapes = self.polished_shapes()
        shapes.advance_holonomy(p, q)
        self.target_meridian_holonomy = shapes.target_holonomy
        self.manifold.set_tetrahedra_shapes(shapes.shapelist)
        self.rough_shapes = ShapeSet(self.manifold, shapes.shapelist)
        self._cache = {}

    def polished_shapes(self, precision=None):
        self._update_precision(precision)
        precision = self.precision
        mangled = "polished_shapes_%s" % precision
        if mangled not in self._cache:
            S = PolishedShapeSet(self.rough_shapes,
                                 self.target_meridian_holonomy, precision)
            self._cache[mangled] = S

        return self._cache[mangled]

    def polished_holonomy(self, precision=None, check=True):
        self._update_precision(precision)
        precision = self.precision
        mangled = "polished_holonomy_%s" % precision
        if mangled not in self._cache:
            G = polished_group(self.manifold,
                               self.polished_shapes().shapelist,
                               precision=precision,
                               fundamental_group_args=self.fundamental_group_args,
                               lift_to_SL2=False, check=check)
            if check:
                if not G.check_representation() < RR(2.0)**(-0.8*precision):
                    raise CheckRepresentationFailed
            self._cache[mangled] = G

        return self._cache[mangled]

    def trace_field_generators(self, precision=None):
        self._update_precision(precision)
        G = self.polished_holonomy()
        return G.trace_field_generators()

    def invariant_trace_field_generators(self, precision=None):
        self._update_precision(precision)
        G = self.polished_holonomy()
        return G.invariant_trace_field_generators()

    def has_real_traces(self, precision=None):
        self._update_precision(precision)
        real_precision = self.precision if self.precision else 15
        max_imaginary_part = max([abs(tr.imag()) for tr in self.trace_field_generators()])
        return  max_imaginary_part < RR(2.0)**(-0.5*real_precision)

    def is_PSU2_rep(self, precision=None):
        if not self.has_real_traces(precision):
            return False
        G = self.polished_holonomy()
        ans, sig, _ = preserves_hermitian_form(G._matrices)
        if ans == False:
            raise ValueError('Rep with real traces does not preserve quad form')
        return sig != 'indefinite'

    def is_PSL2R_rep(self, precision=None):
        if not self.has_real_traces(precision):
            return False
        try:
            G = self.polished_holonomy()
            ans, sig, _ = preserves_hermitian_form(G._matrices)
            if ans == False:
                raise ValueError('Rep with real traces does not preserve quad form')
            return sig != 'definite'
        except:
            pass
        return False

    def is_reducible(self, precision=None):
        G = self.polished_holonomy(precision)
        gens = G.generators()
        real_precision = self.precision if self.precision else 15
        epsilon = RR(2.0)**(-0.9*real_precision)
        for i, g in enumerate(gens):
            a, A = G.SL2C(g), G.SL2C(g.upper())
            for h in gens[i+1:]:
                b, B = G.SL2C(h), G.SL2C(h.upper())
                if abs((a*b*A*B).trace() - 2) > epsilon:
                    return False
        return True

    def really_comes_from_filling(self, precision=None):
        G = self.polished_holonomy(precision)
        return G.check_representation() < RR(2.0)**(-0.8*self.precision)

    def peripheral_curves(self):
        M = self.manifold
        #   Warning: Changing the fundamental group args from the
        #   default (False, False, False) can lead to cases where the
        #   saved meridian word is expressed in terms of different
        #   generators than are actually being used.  Something like
        #   this code might be necessary ...
        #
        #        if False in M.cusp_info('is_complete'):
        #            M = M.copy()
        #            M.dehn_fill([(0,0) for n in range(M.num_cusps())])
        return M.fundamental_group(*self.fundamental_group_args).peripheral_curves()

    def meridian(self):
        return self.peripheral_curves()[0][0]

    def generators(self):
        return self.polished_holonomy().generators()

    def relators(self):
        return self.polished_holonomy().relators()

    def name(self):
        return repr(self.manifold)

    def coboundary_1_matrix(self):
        gens, rels = self.generators(), self.relators()
        # coP maps Free(gens)* -> Free(rels)*
        coP = [[r.count(g) - r.count(g.swapcase()) for g in gens] for r in rels]
        entries = list(chain(*coP))
        return coboundary_matrix(len(rels), len(gens), entries)

    def H2(self):
        """
        Computes H^2(G; Z) *assuming* d_3 : C_3 -> C_2 is the
        zero map.
        """
        if not 'smith_form' in self._cache:
            self._cache['smith_form'] = smith_normal_form(self.coboundary_1_matrix())
        D = self._cache['smith_form'][0] # D = U*coP*V
        ed = elementary_divisors(D)
        ans = [d for d in ed if d != 1]
        return ans

    def has_2_torsion_in_H2(self):
        H2 = self.H2()
        return len([c for c in H2 if c != 0 and c % 2 == 0]) > 0

    def class_in_H2(self, cocycle):
        self.H2()
        D, U = self._cache['smith_form'][:2]
        # U rewrites relators in the smith basis
        ed = elementary_divisors(D)
        co = pari(cocycle).mattranspose()
        ans = []
        coeffs = list(U*co[:][0])
        for c, d in zip(coeffs, list(ed)):
            if d != 1:
                a = c if d == 0 else c % d
                ans.append(a)
        return ans

    def matrix_field(self):
        return self.trace_field_generators()[0].parent()

    def __call__(self, word):
        return self.polished_holonomy().SL2C(word)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

