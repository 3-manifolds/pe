# -*- coding: utf-8 -*-
"""
Define the Elevation class, and its children CircleElevation and LineElevation.
"""
from __future__ import print_function
from collections import defaultdict
from numpy import array, dot, matrix, log, exp, pi, sqrt, zeros
from random import randint
from .gluing import Glunomial, GluingSystem
from .fiber import Fiber
from .fibrator import Fibrator
from .input import user_input
from .plot import Plot
from .shape import PolishedShapeSet, U1Q
from sage.all import ComplexField
import os, sys, time

class Elevation(object):
    """
    Base class for elevations.

    An Elevation is a family of fibers for the meridian holonomy map, obtained
    by taking all lifts of a path (the R-path) in the complex plane.  The lifts
    are sampled at N points, where the value of N is specified by the keyword
    argument *order*.

    The construction begins by using PHC (via the Fibrator class) to find a
    single fiber over the base point ξ_0 in the R-path having index N_0.  (Thus
    the base point need not be an endpoint.)  The value of N_0 can be specified
    with the keyword argument *base_index*.  The default behavior is to choose
    N_0 at random.

    Once a base Fiber has been constructed, it is transported along the path in
    both directions using Newton's method, to construct the full family of
    Fibers.

    An Elevation can be tightened, which means transporting each fiber lying
    over a point on the R-path to one lying over a point on a nearby path, the
    T-path. The idea is that the R-path should be chosen generically, while the
    T-path is non-generic and hence more likely to contain singularities.  For
    example the R-path may travel along a circle of arbitrary radius R close to
    one and the T-path might travel around the unit circle.  Singularities are
    expected on the T-path, and may prevent the transport. Such failures are
    reported on the console and then ignored.
    """
    def __init__(self, manifold, order=128, msg= '', base_dir=None,
                 hint_dir='hints', ignore_saved=False, phc_rescue=False,
                 verbose=True):
        self.manifold = manifold
        self.base_dir = base_dir
        self.hint_dir = hint_dir
        self.order = order
        self.phc_rescue = phc_rescue
        self.verbose = verbose
        self.hp_manifold = self.manifold.high_precision()
        self.dim = manifold.num_tetrahedra()
        self.gluing_system = GluingSystem(manifold)
        self.betti2 = [c % 2 for c in manifold.homology().coefficients].count(0)
        self._check_dir(base_dir, 'I need a directory for storing base fibers.')
        if not ignore_saved:
            saved_data = self._get_saved_data()
        else:
            saved_data = {}
        # Pre-initialize the list of fibers by just inserting an integer for
        # each fiber.  If the fiber construction fails, this can be detected by
        # isinstance(fiber, int)
        self.T_fibers = list(range(order))
        self.R_fibers = list(range(order))
        self._set_paths()
        target = self._get_fibrator_target(saved_data)
        shapes = saved_data.get('shapes', None)
        self.fibrator = Fibrator(manifold, target=target, shapes=shapes, base_dir=base_dir)
        self.base_fiber = base_fiber = self.fibrator()
        arg = log(base_fiber.H_meridian).imag%(2*pi)
        self.base_index = (self.order - int(arg*self.order/(2*pi)))%self.order
        if not base_fiber.is_finite():
            for n, s in enumerate(base_fiber.shapes):
                if s.is_degenerate():
                    print(n, s)
            raise RuntimeError('The base fiber contains Tillmann points.')
        self.degree = len(base_fiber)
        self._print('Order is %d; Degree is %s.'%(self.order, self.degree))
        if msg:
            self._print(msg)
        self.elevate()
        self._finalize()

    def _set_paths():
        raise ValueError('Only subclasses of Elevation can be instantiated.')

    def _get_saved_data(self):
        raise ValueError('Only subclasses of Elevation can be instantiated.')

    def _get_fibrator_target():
        raise ValueError('Only subclasses of Elevation can be instantiated.')

    def _finalize():
        raise ValueError('Only subclasses of Elevation can be instantiated.')

    def __call__(self, Z):
        return array([F(Z) for F in self.glunomials])

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @staticmethod
    def _check_dir(directory, message=''):
        if not os.path.exists(directory):
            cwd = os.path.abspath(os.path.curdir)
            newdir = os.path.join(cwd, directory)
            print('\n'+ message)
            response = user_input("May I create a directory %s?(Y/n)"%newdir)
            if response and response.lower()[0] != 'y':
                sys.exit(0)
            print()
            os.mkdir(newdir)

    def __getitem__(self, index):
        """
        If passed an int, return the T_fiber with that index.  If passed a tuple
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

    def save_hint(self, basename=None, directory=None, extra_options={}):
        """
        Save the settings used to compute this variety.
        """
        if directory == None:
            directory = self.hint_dir
        self._check_dir(directory, 'I need a directory for storing hint files.')
        if basename == None:
            basename = self.manifold.name()
        hintfile_name = os.path.join(directory, basename + '.hint')
        hint_dict = {"manifold": "%s"%self.manifold,
                     "radius" : self.radius,
                     "order" : self.order,
                     }
        hint_dict.update(extra_options)
        str_rep = str(hint_dict).replace(
                '{', '{\n').replace(
                ', ', ',\n').replace(
                '}', '\n}\n')
        with open(hintfile_name, 'w') as hintfile:
            hintfile.write('hint=' + str_rep)

    def transport(self, fiber, target, allow_collision=False, debug=False,
                  fail_quietly=False):
        """
        Transport this fiber to a different target holonomy.  If the resulting
        fiber has a collision, try jiggling the path.
        """
        shapes = []
        failed_points = set()
        for m, shape in enumerate(fiber.shapes):
            if debug:
                print("transport: shape ", len(shapes))
            Zn, failures = self.gluing_system.track(shape.array, target, debug=debug,
                                                   fail_quietly=fail_quietly)
            shapes.append(Zn)
            if failures:
                #print('At point %d: '%m + msg)
                failed_points.add(m)
        result = Fiber(self.manifold, target, shapes=shapes)
        if result.collision() and not allow_collision:
            print("Perturbing the path.")
            result, more_failures = self.retransport(fiber, target, debug=debug,
                                           fail_quietly=fail_quietly)
            failed_points |= more_failures
        return result, failed_points

    def elevate(self):
        """
        Construct the elevation of the R-path.
        """
        self._print('Elevating the R-path', end=' ')
        start = time.time()
        path = self.R_path
        base = self.base_index
        self._print('with base index %s.'%base)
        self._print(' %-5s\r'%base, end='')
        try:
            # Move to the R-path, if necessary.
            self.R_fibers[base], failed = self.transport(self.base_fiber, path[base])
            for n in range(base+1, self.order):
                self._print(' %-5s\r'%n, end='')
                sys.stdout.flush()
                self.R_lift_step(n, 1)
            for n in range(base-1, -1, -1):
                self._print(' %-5s\r'%n, end='')
                sys.stdout.flush()
                self.R_lift_step(n, -1)
            self._print('Tracked in %.2f seconds.'%(time.time() - start))
            self.R_longitude_holos, self.R_longitude_evs, self.R_choices = self.longidata(
                self.R_fibers)
            self.failed = False
        except Exception as e:
            self.failed = True
            raise(e)

    def R_lift_step(self, n, step):
        previous_fiber = self.R_fibers[n-step]
        try:
            F, failed = self.transport(previous_fiber, self.R_path[n])
            self.R_fibers[n] = F
        except Exception as e:
            self._print('\nFailed to compute fiber %s.'%n)
            print(e)
            if self.phc_rescue:
                self._print('Attempting to compute fiber from scratch with PHC ... ', end='')
                F = self.fibrator.PHC_compute_fiber(self.R_path[n])
                self._print('done.')
                F.match_to(previous_fiber)
                self.R_fibers[n] = F
            else:
                raise e
        if not F.is_finite():
            self._print('Degenerate shape! ')

    def tighten(self, path=None):
        """
        Transport each fiber over a point on the R-path to the fiber over
        the corresponding point on the T-path.
        """
        if path is None:
            path = self.T_path
        self._tighten_msg()
        self.tighten_failures = defaultdict(set)
        for n in range(self.order):
            self._print(' %-5s\r'%n, end='')
            sys.stdout.flush()
            failed_points = set()
            try:
                self.T_fibers[n], failed = self.transport(
                    self.R_fibers[n], path[n],
                    allow_collision=True, fail_quietly=True)
                failed_points |= failed
            except Exception as e:
                self._print('Failed to tighten fiber %d.'%n)
            if failed_points:
                bad_points = sorted(list(failed_points))
                self._print('Tightening failures in fiber %d: %s'%(n, bad_points))
                for m in bad_points:
                    self.tighten_failures[m].add(n)
        try:
            self.T_longitude_holos, self.T_longitude_evs, self.T_choices = (
                self.longidata(self.T_fibers))
        except Exception as e:
            self._print('Failed to compute longitude holonomies: %s'%e)

    def longidata(self, fiber_list):
        """
        Compute the longitude holonomies and eigenvalues at each point in each
        fiber in a list of fibers.  We allow the input list of fibers to
        contain placeholders for failed computations.  (A placeholder is
        any object which is not an instance of Fiber.)  The fibers must all
        have degree == self.degree.

        The method produces two lists of lists.  Each outer list has length
        self.degree and each inner list has length self.order, and contains
        a None value for each index where the input fiber list contained a
        placeholder.

        Finding the longitude means choosing a square root λ of the
        longitude holonomy.  As long as the trace of the longitude is
        nonzero there is a unique choice of sign such that λ + 1/λ equals
        the trace.  However, computing the trace of the longitude involves
        constructing a PSL(2,C) representation and lifting it to SL(2,C).
        This is an expensive computation.  So for efficiency we do this
        only at the first computed fiber and for subsequent fibers we
        choose the sign which makes the longitude eigenvalue closer to the
        value computed for the previous fiber.  Note, however, that this
        may fail if the order is chosen to be too small.
        """

        self._print('Computing longitude holonomies and eigenvalues ... ')
        # Compute holonomies with None values where the fiber is a placeholder.
        L_nomial = self.gluing_system.L_nomial
        longitude_holonomies = [
            [L_nomial(f.shapes[m].array) if isinstance(f, Fiber) else None
             for f in fiber_list]
            for m in range(self.degree)]
        # Find the first fiber which has actually been computed.
        index = 0
        while not isinstance(fiber_list[index], Fiber):
            index += 1
        self._print('Using index %d to determine signs.'%index)
        longitude_traces = self.find_longitude_traces(fiber_list[index])
        longitude_eigenvalues = []
        choices = []
        for m, L in enumerate(longitude_holonomies):
            # Choose the sign for the eigenvalue at the starting fiber
            tr = longitude_traces[m]
            e = sqrt(L[index])
            E = [None]*index
            if abs(e + 1/e - tr) > abs(e + 1/e + tr):
                e = -e
            E.append(e)
            Echoices = [self.ev_choice(e)]
            previous = e
            # From here on, choose the sign which makes this eigenvalue closer
            # to the previous eigenvalue.  For this to work correctly the order
            # needs to be sufficiently large.
            for holo in L[index+1:]:
                if holo is None:
                    E.append(None)
                    Echoices.append(None)
                    continue
                e = sqrt(holo)
                if abs(e - previous) > abs(e + previous):
                    e = -e
                E.append(e)
                Echoices.append(self.ev_choice(e))
                previous = e
            longitude_eigenvalues.append(E)
            choices.append(Echoices)
        return longitude_holonomies, longitude_eigenvalues, choices

    def ev_choice(self, ev):
        """
        Compute some numerical information that characterizes which square root was
        chosen to be the eigenvalue.  It is used when computing high precision
        eigenvalues.
        """
        absreal, absimag = abs(ev.real), abs(ev.imag)
        if absreal >= absimag:
            part = 0
            pos = ev.real > 0
        else:
            part = 1
            pos = ev.imag > 0
        return part, pos

    def set_sign(self, choice, ev):
        """
        Determine the sign of a high precision eigenvalue using the data provided by
        ev_choice.  The parameter ev is assumed to be a sage ComplexNumber.
        """
        part, pos = choice
        if part == 0:
            if ev.real() > 0:
                return ev if pos else -ev
            else:
                return -ev if pos else ev
        else:
            if ev.imag() > 0:
                return ev if pos else -ev
            else:
                return -ev if pos else ev

    def polish_R_longitude_vals(self, precision=196):
        R = ComplexField(precision)
        L_holo = self.gluing_system.L_nomial
        # The minus sign is because the FFT path is oriented clockwise.
        path = [R(self.radius)*U1Q(-n, self.order, precision=precision)
                  for n in range(self.order)]
        holos = []
        evs = []
        self._print('Polishing shapes to %d bits precision:'%precision)
        for m in range(self.degree):
            row = []
            for n in range(self.order):
                self._print('\rlift %d %d   '%(m,n), end='')
                r = self.R_fibers[n].shapes[m]
                s = PolishedShapeSet(rough_shapes=r,
                                     target_holonomy=path[n],
                                     precision=precision)
                row.append(L_holo(array(list(s))))
            holos.append(row)
            self._print('\r', end='')
            row = [sqrt(z) for z in row]
            row = [self.set_sign(choice, z) for choice, z in zip(self.R_choices[m], row)]
            evs.append(row)
        self.polished_R_path = path
        self.polished_R_longitude_holos = holos
        self.polished_R_longitude_evs = evs

    def volumes(self, fiber_list=None):
        """Return a list of the volumes of all the characters in a list of fibers."""
        if fiber_list is None:
            fiber_list = self.T_fibers
        volumes = [[] for n in range(self.degree)]
        for fiber in fiber_list:
            if isinstance(fiber, int):
                for n in range(self.degree):
                    volumes[n].append(None)
            else:
                for n, shape in enumerate(fiber.shapes):
                    a = shape.array
                    self.manifold.set_tetrahedra_shapes(a, a, fillings=[(0, 0)])
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
        preimage of the R-path.
        """
        Plot([[complex(x) for x in track] for track in self.R_longitude_evs])

    def show_T_longitude_evs(self):
        """
        Display a plot of the longitude eigenvalues on the meridian
        preimage of the T-path.
        """
        Plot([[complex(x) for x in track] for track in self.T_longitude_evs])

class CircleElevation(Elevation):
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
    def __init__(self, manifold, radius=1.02, tight_radius=1.0, **kwargs):
        self.radius, self.tight_radius = radius, tight_radius
        if 'msg' not in kwargs:
            kwargs['msg'] = 'Using: radius=%g; tight_radius= %g'%(
                self.radius, self.tight_radius)
        super(CircleElevation, self).__init__(manifold, **kwargs)

    def _set_paths(self):
        order = self.order
        Darg = 2*pi/order
        # The minus sign is for consistency with the sign convention
        # used in FFT algorithms
        self.T_path = [self.tight_radius*exp(-n*Darg*1j) for n in range(order)]
        self.R_path = [self.radius*exp(-n*Darg*1j) for n in range(order)]

    def _get_saved_data(self):
        base_fiber_file=os.path.join(self.base_dir, self.manifold.name()+'.base')
        try:
            with open(base_fiber_file) as datafile:
                data = eval(datafile.read())
            self._print('Loaded the base fiber from %s'%base_fiber_file)
        except IOError:
            data = {}
        return data

    def _get_fibrator_target(self, saved_data):
        target = saved_data.get('H_meridian', None)
        target_arg = log(target).imag if target else None
        if target_arg:
            target = self.radius*exp(target_arg*1j)
        else:
            base_index = randint(0, self.order-2)
            self._print('Choosing random base index: %d'%base_index)
            target = self.radius*exp(-2*pi*1j*base_index/self.order)
        return target

    def _finalize(self):
        """
        Checks that the lifted paths close up to form simple curves.
        """
        path = self.R_path
        try:
            self.last_R_fiber, failed = self.transport(self.R_fibers[-1], path[0])
            self._print('Polishing the end fibers ... ')
            self.R_fibers[0].polish()
            self.last_R_fiber.polish()
            self._print('Checking for completeness ... ')
            if not self.last_R_fiber == self.R_fibers[0]:
                self._print('The end fibers did not agree!')
                self._print('This probably indicates that the base fiber is incorrect.')
            else:
                self._print('OK. Lifts close up.')
        except Exception as e:
            self._print('Could not check for completeness.')
            self._print(('%s')%e)

    def _tighten_msg(self):
        self._print('Tightening the circle to radius %g'%self.tight_radius)

    def retransport(self, fiber, target, debug=False, fail_quietly=False):
        """
        Transport this fiber to a different target holonomy following a
        jiggled path which first expands the radius, then advances the
        argument, then reduces the radius. If the resulting fiber has
        a collision, raise an exception.
        """
        for T in (1.01*fiber.H_meridian, 1.01*target, target):
            print('Transporting to %s.'%T)
            shapes = []
            failed_points = set()
            for m, shape in enumerate(fiber.shapes):
                Zn, msg = self.gluing_system.track(shape.array, T, debug=debug,
                                                       fail_quietly=fail_quietly)
                shapes.append(Zn)
                if msg:
                    failed_points.add(m)
            result = Fiber(self.manifold, target, shapes=shapes)
            if result.collision():
                raise ValueError('The collision recurred.  Perturbation failed.')
        return result, failed_points

class LineElevation(Elevation):
    """
    A family of fibers for the meridian holonomy map, lying above the
    unit interval on the real axis.
    """
    def __init__(self, manifold, offset=0.02, **kwargs):
        self.offset = offset
        if 'msg' not in kwargs:
            kwargs['msg'] = 'Using offset=%g'%self.offset
        kwargs['base_dir'] = 'PR_base_fibers'
        super(LineElevation, self).__init__(manifold, **kwargs)

    def _set_paths(self):
        order = self.order
        dx = 1.0/(order + 1) # Don't include the ideal point at M=0
        self.T_path = [(1.0 - n*dx) for n in range(order)]
        self.R_path = [(1.0 - n*dx) + self.offset*1j for n in range(order)]

    def _get_saved_data(self):
        return {}

    def _get_fibrator_target(self, saved_data):
        base_index = randint(0, self.order-2)
        self._print('Choosing random base index: %d'%base_index)
        target = self.R_path[base_index]
        return target

    def _finalize(self):
        """
        Checks that the lifted paths close up to form simple curves.
        """
        pass

    def _tighten_msg(self):
        self._print('Tightening to the real axis.')
        
    def retransport(self, fiber, target, debug=False, fail_quietly=False):
        """
        Transport this fiber to a different target holonomy following a
        jiggled path which first expands the radius, then advances the
        argument, then reduces the radius. If the resulting fiber has
        a collision, raise an exception.
        """
        for T in (fiber.H_meridian + 0.1*1j, target + 0.1*1j, target):
            print('Transporting to %s.'%T)
            shapes = []
            failed_points = set()
            for m, shape in enumerate(fiber.shapes):
                Zn, msg = self.gluing_system.track(shape.array, T, debug=debug,
                                                       fail_quietly=fail_quietly)
                shapes.append(Zn)
                if msg:
                    failed_points.add(m)
            result = Fiber(self.manifold, target, shapes=shapes)
            if result.collision():
                raise ValueError('The collision recurred.  Perturbation failed.')
        return result, failed_points

def solve_mod2_system(the_matrix, rhs):
    """Mod 2 linear algebra - used for lifting reps to SL2C"""
    M, N = the_matrix.shape
    A = zeros((M, N+1), 'i')
    A[:, :-1] = the_matrix
    A[:, -1] = rhs
    S = zeros(N, 'i')
    P = []
    R = list(range(M))
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

