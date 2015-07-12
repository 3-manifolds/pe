from .plot import MatplotPlot as Plot
from .real_reps import (PSL2RRepOf3ManifoldGroup, translation_amount,
                        CouldNotConjugateIntoPSL2R)
from .shape import U1Q
from .euler import euler_cocycle_of_relation, PSL2RtildeElement, LiftedFreeGroupRep
from snappy import CensusKnots
from snappy.snap.polished_reps import MapToFreeAbelianization

from sage.all import RealField, ComplexField, ZZ, log, pi, vector, matrix
from snappy.snap.nsagetools import hyperbolic_torsion


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

class SL2RLifter:
    def __init__(self, V):
        self.elevation = H = V.elevation
        self.degree = H.degree
        self.order = H.order
        self.manifold = V.manifold
        self.set_peripheral_info()
        self.find_shapes()
        print 'lifting reps'
        self.find_reps()
        print 'computing translations'
        self.find_translation_arcs()

    def set_peripheral_info(self):
        G = self.manifold.fundamental_group()
        phi = MapToFreeAbelianization(G)
        m, l = [phi(w)[0] for w in G.peripheral_curves()[0]]
        if m < 0:
            m, l = -m, -l
        self.m_abelian, self.l_abelian = m, l

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
                        current_arc.append( ((s,n), H.T_fibers[n].shapes[s]) )
                    else:
                        current_arc = [ ((s,n), H.T_fibers[n].shapes[s]) ]
                else:
                    if current_arc:
                        if len(current_arc) > 1:
                            self.SL2R_arcs.append(current_arc)
                        current_arc = None
            if current_arc and len(current_arc) > 1:
                self.SL2R_arcs.append(current_arc)

    def find_reps(self):
        self.SL2R_rep_arcs = []
        for arc in self.SL2R_arcs:
            reps = []
            for sn,  S in arc:
                s, n = sn
                # Skipping reps which send the meridian to a parabolic
                # makes for more beautiful pictures, due to the ambiguity
                # of the translation length.  (Is it 0 or 1?)
                if n == 0:
                    continue
                target = U1Q(-n, self.order, precision=1000)
                try:
                    rho = PSL2RRepOf3ManifoldGroup(
                        self.elevation.manifold,
                        target,
                        S,
                        precision=1000)
                    if rho.polished_holonomy().check_representation() < 1.0e-100:
                        reps.append( (sn, rho) )
                except CouldNotConjugateIntoPSL2R:
                    print "Skipping rep probably misclassified as PSL(2,R)"
            if len(reps) > 1: 
                self.SL2R_rep_arcs.append(reps)

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
                    continue 
                try:
                    P = ( float(translation_amount(rho_til(meridian))),
                          float(translation_amount(rho_til(longitude))) )
                    while P[0] < 0:
                        P = (P[0] + self.m_abelian, P[1] + self.l_abelian)
                    while P[0] > self.m_abelian:
                        P = (P[0] - self.m_abelian, P[1] - self.l_abelian)
                    translations.append(P)
                    self.translation_dict[sn] = P
                    rho.translations = P
                except AssertionError:
                    pass
            self.translation_arcs.append(translations)

    def show(self, add_lines=False):
        plotlist = [ [complex(x,y) for x, y in arc] for arc in self.translation_arcs ]
        self.plot = Plot(plotlist, title=self.manifold.name())
        if add_lines:
            self.draw_line(self.manifold.homological_longitude(), color='green')
            for edge in self.l_space_edges():
                self.draw_line(edge, color='red')

    def show_slopes(self):
        M = self.elevation.manifold.copy()
        plotlist = []
        for arc in self.SL2R_arcs:
            slopes = []
            for sn,  S in arc:
                M.set_tetrahedra_shapes(S, S, [(0,0)])
                Hm, Hl = M.cusp_info('holonomies')[0]
                if Hm.imag != 0:
                    slopes.append(float(-Hl.imag/Hm.imag))
                elif len(slopes) > 1:
                    slopes.append(None)
            plotlist.append(slopes)
        self.slope_plot = Plot(plotlist)

    def draw_line(self, curve_on_torus, **kwargs):
        ax = self.plot.figure.axis
        x = ax.get_xlim()[1]
        y = ax.get_ylim()[1]
        a, b = curve_on_torus
        if b != 0:
            ax.plot( (0, x), (0, -a*x/b), **kwargs)
        else:
            ax.plot( (0, 0), (0, -a*5), **kwargs)
        self.plot.figure.draw()

    def l_space_edges(self):
        M = self.manifold
        K = CensusKnots.identify(M)
        if not K:
            return []
        A = M.is_isometric_to(K, True)[0].cusp_maps()[0]
        A = matrix(ZZ,  [[A[0,0], A[0,1]], [A[1,0], A[1,1]]])
        Ainv = A**(-1)
        X = hyperbolic_torsion(M, bits_prec=1000).degree()/2
        l_space_edges = [vector(ZZ, (X, -1)), vector(ZZ, (X,1))]
        return [Ainv*v for v in l_space_edges]

class SL2RLifterHomological(SL2RLifter):
    def __init__(self, V):
        SL2RLifter.__init__(self, V) 
    

def lifted_slope(M,  target_meridian_holonomy_arg, shapes):
    RR = RealField()
    target = RR(target_meridian_holonomy_arg)
    rho = PSL2CRepOf3ManifoldGroup(M, target, rough_shapes=shapes, precision=1000)
    assert rho.polished_holonomy().check_representation() < 1.0e-100
    rho_real = PSL2RRepOf3ManifoldGroup(rho)
    meridian, longitude = rho.polished_holonomy().peripheral_curves()[0]
    rho_tilde = lift_on_cusped_manifold(rho_real)
    return ( -translation_amount(rho_tilde(longitude)) /
             translation_amount(rho_tilde(meridian)) )

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
    assert check_slope(H,low,s) < target_slope < check_slope(H,high,s)
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
    rels = rho.relators()[:rel_cutoff ]
    euler_cocycle = [euler_cocycle_of_relation(rho, R) for R in rels]
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
            return 
    shifts = (-k)[1:]
    good_lifts = [PSL2RtildeElement(rho(g), s)
                  for g, s in zip(rho.generators(), shifts)]
    rho_til= LiftedFreeGroupRep(rho, good_lifts)
    return rho_til


