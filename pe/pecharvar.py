# -*- coding: utf-8 -*-
"""
Define the main class PECharVariety and the helper CircleElevation.

A PECharVariety object represents a Peripherally Elliptic Character
Variety.  Each PECHarVariety oject manages two CircleElevation
objects, which represent a family of fibers for the meridian holonomy
map lying above a circle in the compex plane.
"""
import time, sys, os
from random import randint
from numpy import arange, array, dot, float64, matrix, log, exp, pi, sqrt, zeros
import snappy
from snappy import Manifold, ManifoldHP
from spherogram.graphs import Graph
from .gluing import Glunomial
from .fiber import Fiber
from .fibrator import Fibrator
from .point import PEPoint
from .shape import PolishedShapeSet, U1Q
from .plot import MatplotPlot as Plot
from .complex_reps import PSL2CRepOf3ManifoldGroup
from .real_reps import PSL2RRepOf3ManifoldGroup

class CircleElevation(object):
    """
    A family of fibers for the meridian holonomy map, lying above the
    points Rξ^m where ξ = e^(2πi/N). The value of N is specified by
    the keyword argument *order* (default 128) and the value of R is
    specified by the keyword argument *radius* (default 1.02).

    The construction begins by using PHC (via the Fibrator class) to
    find a single fiber over the point ξ_0 = Re^(2πi/N_0).  The value
    of N_0 can be specified with the keyword argument *base_index*.
    The default behavior is to choose N_0 at random.

    Once a base Fiber has been constructed, it is transported around
    the circle, using Newton's method, to construct the full family of
    Fibers.

    A CircleElevation can be tightened, which means radially
    transporting each fiber lying over a point on the R-circle to one
    lying over a point on the unit circle. Singularities are common on
    the unit circle, and may prevent the transport. Such failures are
    reported on the console and then ignored.
    """
    def __init__(self, manifold, order=128, radius=1.02, target_arg=None,
                 base_fiber_file=None):
        self.order = order
        self.radius = radius
        self.manifold = manifold
        self.hp_manifold = self.manifold.high_precision()
        self.betti2 = [c % 2 for c in manifold.homology().coefficients].count(0)
        Darg = 2*pi/order
        # The minus sign is for consistency with the sign convention
        # used by numpy.fft
        self.R_circle = [radius*exp(-n*Darg*1j) for n in range(self.order)]
        if base_fiber_file and os.path.exists(base_fiber_file):
            target = None
        else:
            if target_arg:
                target = radius*exp(target_arg)
            else:
                base_index = randint(0, order-1)
                print 'Choosing random base index: %d'%base_index
                target = radius*exp(-2*pi*1j*base_index/self.order)
        self.fibrator = Fibrator(manifold, target=target,
                                 fiber_file=base_fiber_file)
        self.base_fiber = base_fiber = self.fibrator()
        arg = log(base_fiber.H_meridian).imag%(2*pi)
        self.base_index = (self.order - int(arg*self.order/(2*pi)))%self.order
        if not base_fiber.is_finite():
            raise RuntimeError('The starting fiber contains Tillmann points.')
        self.degree = len(base_fiber)
        print 'Degree is %s.'%self.degree
        # pre-initialize by just inserting an integer for each fiber
        # if the fiber construction fails, this can be detected by
        # isinstance(fiber, int)
        self.R_fibers = range(order)
        self.T_fibers = range(order)
        self.dim = manifold.num_tetrahedra()
        self.rhs = []
        eqns = manifold.gluing_equations('rect')
        self.glunomials = [Glunomial(A, B, c) for A, B, c in eqns[:-3]]
        self.rhs = [1.0]*(len(eqns) - 3)
        self.M_holo, self.L_holo = [Glunomial(A, B, c) for A, B, c in eqns[-2:]]
        self.glunomials.append(self.M_holo)
        self.track_satellite()
        self.R_longitude_holos, self.R_longitude_evs = self.longidata(self.R_fibers)

    def __call__(self, Z):
        return array([F(Z) for F in self.glunomials])

    def __getitem__(self, index):
        """If passed an int, return the T_fiber with that index.  If passed a tuple
        (fiber_index, shape_index) return the associated ShapeSet.
        """
        try:
            fiber_index = int(index)
            return self.T_fibers[fiber_index]
        except TypeError:
            try:
                fiber_index, shape_index = index
                return self.T_fibers[fiber_index].shapes[shape_index]
            except ValueError:
                raise IndexError('Syntax: V[fiber_index, shape_index]')

    def track_satellite(self):
        """
        Construct the fibers over the circle of radius R.
        """
        print 'Tracking the satellite at radius %s ...'%self.radius
        start = time.time()
        circle = self.R_circle
        base = self.base_index
        print 'Base index is %s'%base
        print ' %-5s\r'%base,
        # Move to the R-circle, if necessary.
        self.R_fibers[base] = self.base_fiber.transport(circle[base])
        for n in xrange(base+1, self.order):
            print ' %-5s\r'%n,
            sys.stdout.flush()
            try:
                self.R_fibers[n] = F = self.R_fibers[n-1].transport(circle[n])
            except Exception as e:
                print '\nfailure at index %d'%n
                raise e
            # self.R_fibers[n].polish()
            if not F.is_finite():
                print '**',
        for n in xrange(base-1, -1, -1):
            print ' %-5s\r'%n,
            sys.stdout.flush()
            try:
                self.R_fibers[n] = F = self.R_fibers[n+1].transport(circle[n])
            except Exception as e:
                print '\nfailure at index %d'%n
                raise e
            if not F.is_finite():
                print '**',
        print
        self.last_R_fiber = self.R_fibers[-1].transport(circle[0])
        print 'Polishing the end fibers ...'
        self.R_fibers[0].polish()
        self.last_R_fiber.polish()
        print 'Checking for completeness ... ',
        if not self.last_R_fiber == self.R_fibers[0]:
            print 'The end fibers did not agree!'
            print 'It might help to use a larger radius, or you might'
            print 'have been unlucky in your choice of base fiber.'
        else:
            print 'OK'
        print 'Tracked in %s seconds.'%(time.time() - start)

    def tighten(self):
        """
        Radially transport each fiber over a point on the R-circle to a
        fiber over a point on the T-circle.
        """
        T = 1.0
        print 'Tightening the circle to radius %s ...'%T
        Darg = 2*pi/self.order
        self.T_circle = circle = [T*exp(-n*Darg*1j) for n in range(self.order)]
        for n in xrange(self.order):
            print ' %-5s\r'%n,
            sys.stdout.flush()
            try:
                self.T_fibers[n] = self.R_fibers[n].transport(circle[n])
            except ValueError:
                print 'Tighten failed at %s'%n
        self.T_longitude_holos, self.T_longitude_evs = self.longidata(
            self.T_fibers)

    def longidata(self, fiber_list):
        """
        Compute the longitude holonomies and eigenvalues at each point in
        each fiber in the argument.  We allow the list to contain
        placeholders for failed computations.  A placeholder is any
        object which is not an instance of Fiber.  The fibers must all
        have degree == self.degree.

        The method produces two lists of lists.  Each outer list has
        length self.degree.  The inner lists contain pairs (n,z) where
        n is the index of the fiber in the input list and z is the
        associated holonomy or eigenvalue.  The integer can be used to
        reconstruct the corresponding meridian holonomy or eigenvalue
        in the case where the list of fibers is the elevation of a
        sampled circle.

        """
        print 'Computing longitude holonomies and eigenvalues.'
        # This crashes if there are bad fibers.
        longitude_holonomies = [
            [(n, self.L_holo(f.shapes[m].array))
             for n, f in enumerate(fiber_list)
             if isinstance(f, Fiber)]
            for m in xrange(self.degree)]
        if isinstance(fiber_list[0], Fiber):
            index = 0
        else:
            index = randint(0, self.order - 1)
            print 'Using %d as the starting index.'%index
        longitude_traces = self.find_longitude_traces(fiber_list[index])
        longitude_eigenvalues = []
        for m, L in enumerate(longitude_holonomies):
            tr = longitude_traces[m]
            n, holo = L[index]
            e = sqrt(holo)
            # Choose the sign for the eigenvalue at the starting fiber
            E = [(n, e) if abs(e + 1/e - tr) < abs(e + 1/e + tr) else (n, -e)]
            # Avoid discontinuities caused by the branch cut used by sqrt
            for n, holo in L[index+1:]:
                e = sqrt(holo)
                E.append((n, e) if abs(e - E[-1][1]) < abs(e + E[-1][1]) else (n, -e))
            if index > 0:
                for n, holo in L[index-1::-1]:
                    e = sqrt(holo)
                    E.insert(0, (n, e) if abs(e - E[0][1]) < abs(e + E[0][1]) else (n, -e))
            longitude_eigenvalues.append(E)
        return longitude_holonomies, longitude_eigenvalues

    def compute_volumes(self, fiber_list):
        """Return a list of the volumes of all the characters in a list of fibers."""
        volumes = [[] for n in range(self.degree)]
        for fiber in fiber_list:
            for n, shape in enumerate(fiber.shapes):
                self.manifold.set_tetrahedra_shapes(shape(), fillings=[(0, 0)])
                volumes[n].append(self.manifold.volume())
        return volumes

    def find_longitude_traces(self, fiber):
        """Compute the longitude traces by lifting to SL(2,C)."""
        # Sage complex numbers do not support attributes .real and .imag :^(((
        trace = lambda rep: complex(rep[0, 0] + rep[1, 1])
        traces = []
        for shape in fiber.shapes:
            a = shape.array
            self.hp_manifold.set_tetrahedra_shapes(a, a, [(0, 0)])
            G = self.hp_manifold.fundamental_group()
            longitude = G.peripheral_curves()[0][1]
            relators = G.relators()
            generators = G.generators()
            M, N = len(relators), G.num_generators()
            A = matrix(zeros((M, N), 'i'))
            L = zeros(N, 'i')
            rhs = zeros(M, 'i')
            for i in range(M):
                for j in range(N):
                    A[i, j] = (relators[i].count(generators[j]) +
                               relators[i].count(generators[j].upper()))%2
                    L[j] = (longitude.count(generators[j]) +
                            longitude.count(generators[j].upper()))%2
                rhs[i] = trace(G.SL2C(relators[i])).real < 0
            S = matrix(solve_mod2_system(A, rhs)).transpose()
            # Paranoia
            if max((A*S - matrix(rhs).transpose())%2) > 0:
                if self.betti2 == 1:
                    raise RuntimeError("Mod 2 solver failed!")
            tr = trace(G.SL2C(longitude))
            if (L*S)%2 != 0:
                tr = -tr
            traces.append(tr)
        return traces

    def show_R_longitude_evs(self):
        """
        Display a plot of the longitude eigenvalues on the meridian
        preimage of the R-circle.
        """
        Plot([[complex(x) for _, x in track] for track in self.R_longitude_evs])

    def show_T_longitude_evs(self):
        """
        Display a plot of the longitude eigenvalues on the meridian
        preimage of the T-circle.
        """
        Plot([[complex(x) for _, x in track] for track in self.T_longitude_evs])

def solve_mod2_system(the_matrix, rhs):
    """Mod 2 linear algebra - used for lifting reps to SL2C"""
    M, N = the_matrix.shape
    A = zeros((M, N+1), 'i')
    A[:, :-1] = the_matrix
    A[:, -1] = rhs
    S = zeros(N, 'i')
    P = []
    R = range(M)
    r = 0
    for j in range(N):
        i = r
        while i < M:
            if A[R[i]][j] != 0:
                break
            i += 1
        if i == M:
            continue
        if i > r:
            R[r], R[i] = R[i], R[r]
        P.insert(0, j)
        for i in range(r+1, M):
            if A[R[i]][j] == 1:
                A[R[i]] = A[R[i]]^A[R[r]]
        r += 1
    i = len(P) - 1
    for j in P:
        S[j] = (A[R[i]][N] - dot(A[R[i]][j+1:-1], S[j+1:]))%2
        i -= 1
    return S


class PEArc(list):
    """
    A list of pillowcase points lying on an arc of the PECharVariety.
    Subclassed here to allow additional attributes.
    """

class PECharVariety(object):
    """Representation of the PE Character Variety of a 3-manifold."""
    def __init__(self, manifold, order=128, radius=1.02,
                 elevation=None, base_dir='PE_base_fibers', hint_dir='hints'):
        if isinstance(manifold, (Manifold, ManifoldHP)):
            self.manifold = manifold
        else:
            self.manifold = Manifold(manifold)
        self.radius = radius
        self.order = order
        self.hint_dir = hint_dir
        if elevation is None:
            self._check_dir(base_dir, 'I need a directory for storing base fibers.')
            self.elevation = CircleElevation(
                self.manifold,
                order=order,
                radius=radius,
                base_fiber_file=os.path.join(
                    base_dir, self.manifold.name()+'.base')
                )
            self.elevation.tighten()
        else:
            self.elevation = elevation

    def __getitem__(self, index):
        """
        Return the indexed fiber or shape from this PECharVariety's elevation.
        """
        return self.elevation[index]

    @staticmethod
    def _check_dir(directory, message=''):
        if not os.path.exists(directory):
            cwd = os.path.abspath(os.path.curdir)
            newdir = os.path.join(cwd, directory)
            print '\n'+ message
            response = raw_input("May I create a directory %s?(Y/n)"%newdir)
            if response and response.lower()[0] != 'y':
                sys.exit(0)
            print
            os.mkdir(newdir)

    def save_hint(self, basename=None, directory=None):
        """Save the settings used to compute this variety."""
        if directory == None:
            directory = self.hint_dir
        self._check_dir(directory, 'I need a directory for storing hint files.')
        if basename == None:
            basename = self.manifold.name()
        hintfile_name = os.path.join(directory, basename + '.hint')
        hintfile = open(hintfile_name, 'w')
        hintfile.write('hint={\n')
        hintfile.write('"manifold" : %s,\n'%self.manifold)
        hintfile.write('"radius" : %f,\n'%self.radius)
        hintfile.write('"order" : %d,\n'%self.order)
        hintfile.write('}\n')
        hintfile.close()

    def build_arcs(self, show_group=False):
        """Find the arcs in the pillowcase projection of this PE Character Variety."""
        self.arcs = []
        self.arc_info = []
        H = self.elevation
        delta_M = -1.0/self.order
        M_args = 0.5 * (arange(self.order, dtype=float64)*delta_M % 1.0)
        for m, track in enumerate(self.elevation.T_longitude_evs):
            arc, info = PEArc(), []
            marker = ''
            for n, ev in track:
                if 0.99999 < abs(ev) < 1.00001:
                    if show_group:
                        shape = H.T_fibers[n].shapes[m]
                        if shape.in_SU2():
                            marker = '.'
                        elif shape.has_real_traces():
                            marker = 'D'
                        else:
                            marker = 'x'
                    L = (log(ev).imag/(2*pi))%1.0
                    if len(arc) > 2:  # don't do this near the corners.
                        last_L = arc[-1].real
                        if last_L > 0.8 and L < 0.2:   # L became > 1
                            length = 1.0 - last_L + L
                            interp = ((1.0-last_L)*M_args[n] + L*M_args[n-1])/length
                            arc.append(PEPoint(1.0, interp, leave_gap=True,
                                               marker=marker))
                            arc.append(PEPoint(0.0, interp))
                        elif last_L < 0.2 and L > 0.8: # L became < 0
                            length = last_L + 1.0 - L
                            interp = (last_L*M_args[n] + (1.0 - L)*M_args[n-1])/length
                            arc.append(PEPoint(0.0, interp, leave_gap=True))
                            arc.append(PEPoint(1.0, interp))
                    arc.append(PEPoint(L, M_args[n], marker=marker, index=(m, n)))
                    info.append((m, n))
                else:
                    if len(arc) > 1:
                        m, n = arc.first_info = info[0]
                        arc.first_shape = self.elevation.T_fibers[n].shapes[m]
                        m, n = arc.last_info = info[-1]
                        arc.last_shape = self.elevation.T_fibers[n].shapes[m]
                        self.arcs.append(arc)
                        self.arc_info.append(info)
                    arc = PEArc()
                    info = []
            if arc:
                # Dumb repetition
                m, n = arc.first_info = info[0]
                arc.first_shape = self.elevation.T_fibers[n].shapes[m]
                m, n = arc.last_info = info[-1]
                arc.last_shape = self.elevation.T_fibers[n].shapes[m]
                self.arcs.append(arc)
                self.arc_info.append(info)
        self.curve_graph = curve_graph = Graph([], range(len(self.arcs)))
        self.add_extrema()
        # build the color dict
        self.colors = {}
        for n, component in enumerate(curve_graph.components()):
            for m in component:
                self.colors[m] = n
        # Clean up endpoints at the corners of the pillowcase.
        for arc in self.arcs:
            try:
                if abs(arc[1] - arc[0]) > 0.25:
                    if arc[0].imag > 0.45 and arc[1].imag < 0.05:
                        arc[0] = arc[0] - 0.5j
                    elif arc[0].imag < 0.05 and arc[1].imag > 0.45:
                        arc[0] = arc[0]+ 0.5j
                    if arc[0].real > 0.9 and arc[1].real < 0.1:
                        arc[0] = arc[0] - 1.0
                    elif arc[0].real < 0.1 and arc[1].real > 0.9:
                        arc[0] = arc[0] + 1.0
                if abs(arc[-1] - arc[-2]) > 0.25:
                    if abs(arc[-2].imag) < 0.05 and arc[-1].imag > 0.45:
                        arc[-1] = arc[-1] - 0.5j
                    elif arc[-2].imag > 0.45 and arc[-1].imag < 0.05:
                        arc[-1] = arc[-1] + 0.5j
                    if arc[-2].real < 0.1 and arc[-1].real > 0.9:
                        arc[-1] = arc[-1] - 1.0
                    elif arc[-2].real > 0.9 and arc[-1].real < 0.1:
                        arc[-1] = arc[-1] + 1.0
            except TypeError:
                pass

    def add_extrema(self):
        """
        The pillowcase arcs, as computed, are monotone decreasing in the
        argument of the meridian.  The actual components of the
        pillowcase picture are unions of these monotone arcs joined at
        critical points of the argument of the meridian.  This method
        attempts to compute which arcs should be joined at maxima or
        minima.
        """
        arcs = list(self.arcs)
        # caps
        arcs.sort(key=lambda x: x[0].imag, reverse=True)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[0].imag == arcs[0][0].imag:
                level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].first_shape.dist(a.first_shape)
                                   for a in level[1:]])
                cap = [level.pop(0), level.pop(distances.argmin())]
                cap.sort(key=lambda a: a[0].real)
                left, right = cap
                if .01 < right[0].imag < .49:
                    join = True
                    if right[1].real > right[0].real and left[1].real < left[0].real:
                        right.insert(0, left[0])
                    elif right[1].real < right[0].real and left[1].real > left[0].real:
                        right.insert(0, PEPoint(1.0, left[0].imag))
                        left.insert(0, PEPoint(0.0, left[0].imag))
                    else:
                        join = False
                    if join:
                        self.curve_graph.add_edge(
                            self.arcs.index(left),
                            self.arcs.index(right))
        # cups
        arcs = list(self.arcs)
        arcs.sort(key=lambda x: x[-1].imag)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[-1].imag == arcs[0][-1].imag:
                level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].last_shape.dist(a.last_shape)
                                   for a in level[1:]])
                cup = [level.pop(0), level.pop(distances.argmin())]
                cup.sort(key=lambda a: a[-1].real)
                left, right = cup
                if 0.01 < right[-1].imag < 0.49:
                    join = True
                    if right[-2].real > right[-1].real and left[-2].real < left[-1].real:
                        right.append(left[-1])
                    elif right[-2].real < right[-1].real and left[-2].real > left[-1].real:
                        right.append(PEPoint(1.0, left[-1].imag))
                        left.append(PEPoint(0.0, left[-1].imag))
                    else:
                        join = False
                    if join:
                        self.curve_graph.add_edge(
                            self.arcs.index(left),
                            self.arcs.index(right))
        return

    def show(self, show_group=False):
        """Plot the pillowcase image of this PE Character Variety."""
        self.build_arcs(show_group)
        Plot(self.arcs,
             limits=((0.0, 1.0), (0.0, 0.5)),
             margins=(0, 0),
             aspect='equal',
             title=self.manifold.name(),
             colors=self.colors,
             extra_lines=[((0.5, 0.5), (0.0, 1.0))],
             extra_line_args={'color':'black', 'linewidth':0.75},
             show_group=show_group)

    def get_rep(self, fiber_index, shape_index, precision=1000, tight=True):
        """
        Return a precise representation, computed to the specified binary
        precision, determined by the shape set with index
        *shape_index* in the fiber with index fiber_index over the
        T_circle. If the optional keyname argument *tight* is set to
        False, the R circle is used instead.

        If the repesentation is a PSL(2,R) rep, the return value from
        this rep will be an object of type PSL2RRepOf3ManifoldGRoup
        class.  Otherwise, the return value will be of type
        PSL2CRepOF3ManifoldGRoup.
        """
        if tight:
            target = U1Q(-fiber_index, self.order, precision=precision)
        else:
            target = self.elevation.R_circle[fiber_index]
        fibers = self.elevation.T_fibers if tight else self.elevation.R_fibers
        rough_shapes = fibers[fiber_index].shapes[shape_index]
        polished_shapes = PolishedShapeSet(rough_shapes, target, precision)
        rho = PSL2CRepOf3ManifoldGroup(self.manifold, target,
                                       rough_shapes, precision)
        if rho.is_PSL2R_rep():
            rho = PSL2RRepOf3ManifoldGroup(rho)
        rho.index = (shape_index, fiber_index)
        rho.rep_type = polished_shapes.rep_type()
        return rho
