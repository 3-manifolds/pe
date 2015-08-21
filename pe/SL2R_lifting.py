from .plot import MatplotPlot as Plot
from .real_reps import (PSL2RRepOf3ManifoldGroup,
                        CouldNotConjugateIntoPSL2R,
                        translation_of_lifted_rotation,
                        meridians_fixing_infinity_and_zero)
from .complex_reps import PSL2CRepOf3ManifoldGroup
from .shape import U1Q
from .euler import PSL2RtildeElement, LiftedFreeGroupRep
from .point import PEPoint
from snappy import CensusKnots
from snappy.snap.polished_reps import MapToFreeAbelianization

from sage.all import RealField, ComplexField, ZZ, log, vector, matrix, xgcd
from snappy.snap.nsagetools import hyperbolic_torsion
from time import time

def in_SL2R(H, f, s):
    shape = H.T_fibers[f].shapes[s]
    ev = H.T_longitude_evs[s][f][1]
    if abs(1.0 - abs(ev)) > .00001:
        return False
    if not shape.has_real_traces():
        return False
    if shape.in_SU2():
        return False
    return True

def l1_dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

class SL2RLifter(object):
    def __init__(self, V, silent=False):
        start = time()
        self.elevation = H = V.elevation
        self.degree = H.degree
        self.order = H.order
        self.manifold = V.manifold
        self.set_peripheral_info()
        if not silent:
            print 'polishing shapes ... ',
        self.find_shapes()
        if not silent:
            now = time()
            print '(%.2f secs)\n'%(now - start), 'lifting reps ... ',
            start = now
        self.find_reps()
        if not silent:
            now = time()
            print '(%.2f secs)\n'%(now - start), 'computing translations ... ',
            start = now
        self.find_translation_arcs()
        if not silent:
            now = time()
            print '(%.2f secs)\n'%(now - start),

    def set_peripheral_info(self):
        G = self.manifold.fundamental_group()
        phi = MapToFreeAbelianization(G)
        m, l = [phi(w)[0] for w in G.peripheral_curves()[0]]
        if m < 0:
            m, l = -m, -l
        self.m_abelian, self.l_abelian = m, l

        # We also want to be able to view things from a more homologically
        # natural point of view.

        hom_l = self.manifold.homological_longitude()
        if abs(hom_l[1]) == 1:
            hom_m = (1, 0)
        else:
            a, b = xgcd(*hom_l)[1:]
            hom_m = (b, -a)
            M = self.manifold.copy()
            M.set_peripheral_curves([hom_m, hom_l])
            cusp = M.cusp_info(0).shape
            assert abs(1 + cusp) > 1 and abs(1 - cusp) > 1
        self.hom_m = hom_m
        self.hom_l = hom_l
        self.hom_m_abelian = abs(self.m_abelian*hom_m[0] + self.l_abelian*hom_m[1])
        self.change_trans_to_hom_framing = matrix([hom_m, hom_l])

        # Moreover, we store two special copies of the meridian for
        # later use.

        self.special_meridians = meridians_fixing_infinity_and_zero(self.manifold)

    def find_shapes(self):
        self.SL2R_arcs = []
        H = self.elevation
        for s in range(self.degree):
            current_arc = None
            for n in range(self.order):
                try:
                    point_is_good = in_SL2R(H, n, s)
                except:
                    point_is_good = False
                if point_is_good:
                    if current_arc:
                        current_arc.append(((s, n), H.T_fibers[n].shapes[s]))
                    else:
                        current_arc = [((s, n), H.T_fibers[n].shapes[s])]
                else:
                    if current_arc:
                        if len(current_arc) > 1:
                            self.SL2R_arcs.append(current_arc)
                        current_arc = None
            if current_arc and len(current_arc) > 1:
                self.SL2R_arcs.append(current_arc)

    def find_reps(self):
        self.SL2R_rep_arcs = []
        self.rep_dict = {}
        for arc in self.SL2R_arcs:
            reps = []
            for sn, S in arc:
                _, n = sn
                target = U1Q(-n, self.order, precision=1000)
                try:
                    rho = PSL2RRepOf3ManifoldGroup(
                        self.elevation.manifold,
                        target,
                        S,
                        precision=1000,
                        special_meridians=self.special_meridians
                    )
                    if rho.polished_holonomy().check_representation() < 1.0e-100:
                        reps.append((sn, rho))
                        self.rep_dict[sn] = rho
                except CouldNotConjugateIntoPSL2R:
                    print "Skipping rep probably misclassified as PSL(2,R)"
            if len(reps) > 1:
                self.SL2R_rep_arcs.append(reps)

        self.correct_rep_endpoints()

    def correct_rep_endpoints(self):
        """
        Deal with minor discontinuities at parabolic endpoints of the rep
        arcs
        """
        for arc in self.SL2R_rep_arcs:
            for i, j in [(0, 1), (-2, -1)]:
                rho0, rho1 = arc[i][1], arc[j][1]
                A0, A1 = rho0('a')[1], rho1('a')[1]
                if (A0 + A1).norm() < (A0 - A1).norm():
                    rho0.flip()

    def find_translation_arcs(self):
        self.translation_arcs = []
        self.translation_dict = {}
        for arc in self.SL2R_rep_arcs:
            translations = []
            for sn, rho in arc:
                rho.translations = None
                meridian, longitude = rho.polished_holonomy().peripheral_curves()[0]
                rho_til = lift_on_cusped_manifold(rho)
                if rho_til is None:
                    print 'No lift!'
                    continue
                try:
                    P = (float(translation_of_lifted_rotation(rho_til(meridian))),
                         float(translation_of_lifted_rotation(rho_til(longitude))))
                    while P[0] < 0:
                        P = (P[0] + self.m_abelian, P[1] + self.l_abelian)
                    while P[0] >= self.m_abelian:
                        P = (P[0] - self.m_abelian, P[1] - self.l_abelian)
                except AssertionError:
                    print "Warning: an assertion failed somewhere"
                translations.append(self._saved_point(P, sn, rho))
            self._fix_translations(translations)
            self.translation_arcs.append(translations)

    def _fix_translations(self, translations):
        """
        Check for reps with parabolic or traceless meridians and adjust their
        translations to make the translation arc continuous.
        """
        fix_list = []
        neighbor = translations[1].tuple()
        for n, P in enumerate(translations):
            p, sn = P.tuple(), P.index
            rho = self.rep_dict[sn]
            meridian_trace = float(rho(rho.meridian()).trace())
            if meridian_trace == 2.0 or meridian_trace == -2.0:
                # Both translations will be (apparently random) integers.
                #print 'fixing parabolic meridian', sn, p, '->',
                fixed = (round(neighbor[0]), round(neighbor[1]))
                #print fixed, neighbor
                fix_list.append((n, self._saved_point(fixed, sn, rho)))
            elif abs(meridian_trace) < 2.0**-100:
                # The translations may have the wrong sign.
                #print 'fixing traceless meridian', sn, p, '->',
                fixed = p
                flipped = (self.m_abelian - p[0], self.l_abelian - p[1])
                if l1_dist(flipped, neighbor) < l1_dist(p, neighbor):
                    fix_list.append((n, self._saved_point(flipped, sn, rho)))
                    #fixed = flipped
                #print fixed, neighbor
            neighbor = p
        for n, fixed in fix_list:
            translations[n] = fixed

    def _saved_point(self, P, sn, rho):
        point = PEPoint(complex(*P), index=sn)
        self.translation_dict[sn] = rho.translations = point
        return point

    def show(self, add_lines=False):
        self.plot = Plot(self.translation_arcs, title=self.manifold.name())
        if add_lines:
            self.draw_line(self.manifold.homological_longitude(), color='green')
            for edge in self.l_space_edges():
                self.draw_line(edge, color='red')

# This puts too many buttons on the graph.
    def show_homological(self):
        A = self.change_trans_to_hom_framing
        m = self.hom_m_abelian
        plotlist = []
        for arc in self.translation_arcs:
            reframed_arc = []
            for t in arc:
                x, y = A*vector((t.real, t.imag))
                while x < 0:
                    x += m
                while x > m:
                    x -= m
                # Primitive way of dealing with crossing fundamental domains
                if reframed_arc and abs(reframed_arc[-1].real - x) > 0.5:
                    plotlist.append(reframed_arc)
                    reframed_arc = []
                reframed_arc.append(PEPoint(complex(x, y), index=t.index))
            plotlist.append(reframed_arc)
        self.plot = Plot(plotlist, title=self.manifold.name() + ' reframed')
        # Draw longitude
        ax = self.plot.figure.axis
        ax.plot((0, 1), (0, 0), color='green')
        self.plot.figure.draw()

    def show_slopes(self):
        M = self.elevation.manifold.copy()
        plotlist = []
        for arc in self.SL2R_arcs:
            slopes = []
            for _, S in arc:
                M.set_tetrahedra_shapes(S, S, [(0, 0)])
                Hm, Hl = M.cusp_info('holonomies')[0]
                if Hm.imag() != 0:
                    slopes.append(float(-Hl.imag()/Hm.imag()))
                elif len(slopes) > 1:
                    slopes.append(None)
            plotlist.append(slopes)
        self.slope_plot = Plot(plotlist)

    def draw_line(self, curve_on_torus, **kwargs):
        ax = self.plot.figure.axis
        x = ax.get_xlim()[1]
        a, b = curve_on_torus
        if b != 0:
            ax.plot((0, x), (0, -a*x/b), **kwargs)
        else:
            ax.plot((0, 0), (0, -a*5), **kwargs)
        self.plot.figure.draw()

    def l_space_edges(self):
        # self.manifold will not have the expected shapes, so make a copy.
        M = self.manifold.copy()
        K = CensusKnots.identify(M)
        if not K:
            return []
        A = M.is_isometric_to(K, True)[0].cusp_maps()[0]
        A = matrix(ZZ, [[A[0, 0], A[0, 1]], [A[1, 0], A[1, 1]]])
        Ainv = A**(-1)
        X = hyperbolic_torsion(M, bits_prec=1000).degree()/2
        l_space_edges = [vector(ZZ, (X, -1)), vector(ZZ, (X, 1))]
        return [Ainv*v for v in l_space_edges]

class SL2RLifterHomological(SL2RLifter):
    def __init__(self, V):
        SL2RLifter.__init__(self, V)


def lifted_slope(M, target_meridian_holonomy_arg, shapes):
    RR = RealField()
    target = RR(target_meridian_holonomy_arg)
    rho = PSL2CRepOf3ManifoldGroup(M, target, rough_shapes=shapes, precision=1000)
    assert rho.polished_holonomy().check_representation() < 1.0e-100
    rho_real = PSL2RRepOf3ManifoldGroup(rho)
    meridian, longitude = rho.polished_holonomy().peripheral_curves()[0]
    rho_tilde = lift_on_cusped_manifold(rho_real)
    return (-translation_of_lifted_rotation(rho_tilde(longitude)) /
            translation_of_lifted_rotation(rho_tilde(meridian)))

def check_slope(H, n, s):
    F = H.T_fibers[n]
    S = F.shapes[s]
    M = H.manifold.copy()
    target = log(F.H_meridian).imag()
    return float(lifted_slope(M, target, S))

def bisection(H, low, high, s, target_slope, epsilon=1.0e-8):
    CC = ComplexField()
    low_fiber = H.T_fibers[low]
    high_fiber = H.T_fibers[high]
    M = H.manifold
    F = H.fibrator
    assert check_slope(H, low, s) < target_slope < check_slope(H, high, s)
    print 'finding:', target_slope
    count = 0
    while count < 100:
        z = (low_fiber.H_meridian + high_fiber.H_meridian)/2
        target_holonomy = z/abs(z)
        target_holonomy_arg = CC(target_holonomy).log().imag()
        new_fiber = F.transport2(low_fiber, complex(target_holonomy))
        shapes = new_fiber.shapes[s]
        new_slope = lifted_slope(M, target_holonomy_arg, shapes)
        if abs(new_slope - target_slope) < epsilon:
            return new_fiber.shapes[s]
        if new_slope < target_slope:
            low_fiber = new_fiber
            print new_slope, 'too low'
        else:
            high_fiber = new_fiber
            print new_slope, 'too high'
        count += 1
        print count
    print 'limit exceeded'
    return new_fiber.shapes[s]

def lift_on_cusped_manifold(rho):
    rel_cutoff = len(rho.generators()) - 1
    rels = rho.relators()[:rel_cutoff]
    euler_cocycle = rho.euler_cocycle_on_relations()
    D = rho.coboundary_1_matrix()[:rel_cutoff]
    M = matrix([euler_cocycle] + D.columns())
    k = M.left_kernel().basis()[0]
    if k[0] != 1:
        # Two reasons we could be here: the euler class isn't zero or
        # the implicit assumption about how left_kernel works is violated.
        # Only the latter is actually worrysome.
        if D.elementary_divisors() == M.transpose().elementary_divisors():
            raise AssertionError('Need better implementation, Nathan')
        else:
            return None, None
    shifts = (-k)[1:]
    good_lifts = [PSL2RtildeElement(rho(g), s)
                  for g, s in zip(rho.generators(), shifts)]
    return LiftedFreeGroupRep(rho, good_lifts)


