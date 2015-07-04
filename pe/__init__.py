# -*- coding: utf-8 -*-

import time, sys, os
from random import randint
import numpy
from numpy import array, matrix, ndarray
from numpy import dot, prod, diag, transpose, zeros, ones, eye
from numpy import log, exp, pi, sqrt, ceil
from numpy import dtype, take, arange, sum, where, vstack, argmin
from numpy import float64
from numpy.linalg import svd, norm, eig, solve, lstsq, matrix_rank
from numpy.fft import ifft
from phc import PolyRing, PHCPoly, PHCSystem, ParametrizedSystem
import snappy
snappy.SnapPy.matrix = matrix
snappy.SnapPyHP.matrix = matrix
from snappy import *
from spherogram.graphs import Graph
from point import PEPoint
from shape import Shapes, PolishedShapes, U1Q
from plot import MatplotPlot as Plot

# Constants for Newton's method
RESIDUAL_BOUND = 1.0E-14
STEPSIZE_BOUND = 1.0E-14
# The numpy type for our complex arrays
DTYPE = dtype('c16')
# Check if we are running in Sage
try:
    import sage
    got_sage = True
except ImportError:
    got_sage = False

class Glunomial:
    """
    A product of powers of linear terms z_i or (1-z_i), as appears on
    the left side of a gluing equation.  These are Laurent monomials;
    powers may be negative.  Instantiate with one of a triple as
    returned by Manifold.gluing_equations('rect').
    """
    def __init__(self, A, B, c):
        self.A, self.B, self.sign = array(A), array(B), float(c)

    def __repr__(self):
        apower = lambda n, p: 'z%d^%s'%(n,p) if p != 1 else 'z%s'%n
        bpower = lambda n, p: '(1-z%d)^%s'%(n,p) if p != 1 else '(1-z%s)'%n
        Apowers = [apower(n, a) for n, a in enumerate(self.A) if a != 0]
        Bpowers = [bpower(n, b) for n, b in enumerate(self.B) if b != 0]
        sign = '' if self.sign == 1.0 else '-'
        return sign + '*'.join(Apowers+Bpowers)

    def __call__(self, Z):
        W = 1 - Z
        try:
            return self.sign*prod(Z**self.A)*prod(W**self.B)
        except ValueError:
            print 'Glunomial evaluation crashed on %s'%self
            print 'A =', self.A
            print 'B =', self.B
            print 'c =', self.sign
            print 'Z =', Z
            raise ValueError

    def gradient(self, Z):
        W = 1 - Z
        return self.sign*prod(Z**self.A)*prod(W**self.B)*(self.A/Z - self.B/W)

class GluingSystem:
    """
    The system of gluing equations for a manifold, with specified
    meridian holonomy.  If the manifold has n tetrahedra, we use the
    first n-1 edge equations, together with the equation for meridian
    holonomy.  The left hand side of each equation is a Laurent monomial
    in z_i and (1-z_i), where z_i are the shape paremeters.  The
    right hand side of the system is [1,...,1,Hm] where Hm is the
    the meridian holonomy (i.e. the first eigenvalue squared).
    """
    def __init__(self, manifold):
        assert manifold.num_cusps() == 1, 'Manifold must be one-cusped.'
        self.manifold = manifold
        eqns = manifold.gluing_equations('rect')
        # drop the last edge equation
        self.glunomials = [Glunomial(A, B, c) for A, B, c in eqns[:-3]]
        self.M_nomial, self.L_nomial = [Glunomial(A, B, c) for A,B,c in eqns[-2:]]
        self.glunomials.append(self.M_nomial)

    def __repr__(self):
        return '\n'.join([str(G) for G in self.glunomials])

    def __call__(self, Z):
        return array([G(Z) for G in self.glunomials])

    def __len__(self):
        return len(self.glunomials)
    
    def jacobian(self, Z):
        return matrix([G.gradient(Z) for G in self.glunomials])

    def M_holonomy(self, Z):
        return complex(self.M_nomial(Z))

    def L_holonomy(self, Z):
        return complex(self.L_nomial(Z))

    def condition(self, Z):
        """
        Return the condition numbers of the Jacobians for the defining
        equations of the gluing curve and of the entire system
        at the point Z.
        """
        U, D, V = svd(self.jacobian(Z)[:-1])
        curve = D[0]/D[-1]
        U, D, V = svd(self.jacobian(Z))
        system = D[0]/D[-1]
        return curve, system
    
    def newton_step(self, Z, M_target):
        """
        Do one iteration of Newton's method, starting at Z and aiming
        to solve G(z) = (1,1,...,M_target).  Returns a triple:
        Z', step_size, residual.  Solves the linear system by 
        LU factorization (not great for nearly singular systems).
        """
        J = self.jacobian(Z)
        target = ones(len(self), dtype=DTYPE)
        target[-1] = M_target
        dZ = solve(J, target - self(Z))
        step_size = norm(dZ)
        Zn = Z + dZ
        return Zn, step_size, max(abs(target - self(Zn)))
    
    def newton_step_ls(self, Z, M_target):
        """
        Do one iteration of Newton's method, starting at Z and aiming
        to solve G(z) = (1,1,...,M_target). Returns a pair:
        dZ, (1,1,...,M_target).  Finds a least squares approcimate
        solution to the linear system.  This method is stable with nearly
        singular systems.
        """
        J = self.jacobian(Z)
        target = ones(len(self), dtype=DTYPE)
        target[-1] = M_target
        error = target - self(Z)
        dZ, residues, rank, sing = lstsq(J, error)
        return dZ, target

    def newton1(self, Z, M_target, debug=False):
        """
        Simple version of Newton's method.  Uses the LU decomposition to
        solve the linear system.  Does not adjust step sizes.
        The iteration is terminated if:
          * the residual does not decrease; or
          * the step size is smaller than 1.0E-15
          * more than 10 iterations have been attempted
        """
        prev_residual = step_size = 1.0E5
        prev_Z, count = Z, 1
        while True:
            Zn, step_size, residual = self.newton_step(prev_Z, M_target)
            if debug: print count, residual, step_size
            if residual > prev_residual:
                return prev_Z, prev_residual
            if (step_size < STEPSIZE_BOUND or
                residual < RESIDUAL_BOUND or
                count > 10):
                return Zn, residual
            prev_Z, prev_residual = Zn, residual
            count += 1

    def newton2(self, Z, M_target, debug=False):
        """
        Fancier version of Newton's method which uses a least squares
        solution to the linear system.  To avoid overshooting, step
        sizes are adjusted by an Armijo rule that successively halves
        the step size.
        The iteration is terminated if:
          * the residual does not decrease; or
          * the step size is smaller than 1.0E-15
          * more than 10 iterations have been attempted
        """
        prev_residual = step_size = 1.0E5
        prev_Z, count = Z, 1
        while True:
            dZ, target = self.newton_step_ls(prev_Z, M_target)
            t = 1.0
            for k in range(5):
                Zn = prev_Z + t*dZ
                residual = max(abs(target - self(Zn)))
                if residual < prev_residual:
                    break
                else:
                    t *= 0.5
            if debug: print 'scaled dZ by %s; residual: %s'%(t, residual)
            if residual > prev_residual:
                if debug: print 'Armijo failed with t=%s'%t
                return prev_Z, prev_residual
            step_size = norm(t*dZ)
            if (step_size < STEPSIZE_BOUND or
                residual < RESIDUAL_BOUND or
                count > 10):
                return Zn, residual
            prev_Z, prev_residual = Zn, residual
            count += 1

    def track(self, Z, M_target, dT=1.0, debug=False):
        """
        Track solutions of the gluing system starting at Z and
        ending at a solution where the meridian holonomy takes the
        specified value.  The path is subdivided into subpaths of
        length determined by dT, which may be further divided if
        convergence problems arise.
        """
        M_start = self(Z)[-1]
        delta = (M_target - M_start)
        T = 0.0
        Zn = Z
        if debug: print 'Z = %s; condition=%s'%(
                Z, [self.condition(x) for x in Z]) 
        # First we try the cheap and easy method
        while T < 1.0:
            T = T+dT
            target = M_start + T*delta
            Zn, residual = self.newton1(Zn, target)
        if residual < 1.0E-14: # What is a good threshold here?
            return Zn
        # If that fails, try taking baby steps.
        if debug: print 'Taking baby steps ...'
        success = 0
        T, dT = 0.0, 0.5*dT
        prev_Z = Z
        while T < 1.0:
            Tn = min(T+dT, 1.0)
            if debug: print 'trying T = %.17f'%Tn
            baby_target = M_start + Tn*delta
            Zn, residual = self.newton2(prev_Z, baby_target, debug=debug)
            if residual < 1.0E-12:
                prev_Z = Zn
                if success > 3:
                    success = 0
                    dT *= 2
                    if debug: print 'Track step increased to %.17f'%dT
                else:
                    success += 1
                T = Tn
            else:
                success = 0
                dT /= 2
                if debug: print 'Track step reduced to %.17f; condition = %s'%(
                        dT,
                        self.condition(prev_Z))
                if dT < 2.0**(-16):
                    raise ValueError('Track failed: step size limit reached.')
        return Zn

class Fiber:
    """A fiber for the rational function [holonomy of the meridian]
    restricted to the curve defined by the gluing system for a
    triangulated cusped manifold.  Can be initialized with a PHCSystem
    and a list of PHCsolutions.

    """
    def __init__(self, manifold, H_meridian, gluing_system=None,
                 PHCsystem=None, shapes=None, tolerance=1.0E-05):
        self.hp_manifold = manifold.high_precision()
        # Here the tolerance is used to determine which of the PHC solutions
        # are at infinity.
        self.H_meridian = H_meridian
        self.tolerance = tolerance
        if shapes:
            self.shapes = [Shapes(self.hp_manifold, S) for S in shapes]
        if gluing_system is None:
            self.gluing_system = GluingSystem(manifold)
        else:
            self.gluing_system=gluing_system
        self.system = PHCsystem
        if self.system:
            self.extract_info()
        
    def extract_info(self):
        N = self.system.num_variables()/2
        self.solutions = self.system.solution_list(tolerance=self.tolerance)
        # We only keep the "X" variables.
        self.shapes = [Shapes(self.hp_manifold, S.point[:N]) for S in self.solutions]

    def __repr__(self):
        return "Fiber(ManifoldHP('%s'),\n%s,\nshapes=%s\n)"%(
            repr(self.hp_manifold),
            repr(self.H_meridian),
            repr([list(x) for x in self.shapes]).replace('],','],\n')
            )
    
    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        return self.shapes[index]
    
    def __eq__(self, other):
        """
        This ignores multiplicities.
        """
        for p in self.shapes:
            if p not in other.shapes:
                return False
        for p in other.shapes:
            if p not in self.shapes:
                return False
        return True
    
    def collision(self):
        for n, p in enumerate(self.shapes):
            for q in self.shapes[n+1:]:
                if p.dist(q) < 1.0E-10:
                    return True
        return False

    def is_finite(self):
        """
        Check if any cross-ratios are 0 or 1
        """
        for p in self.shapes:
            if p.is_degenerate():
                return False
        return True
            
    def details(self):
        # broken if not instantiated with a PHCsystem
        for n, s in enumerate(self.solutions):
            print 'solution #%s:'%n
            print s

    def residuals(self):
        # broken if not instantiated with a PHCsystem
        for n, s in enumerate(self.solutions):
            print n, s.res 

    def polish(self):
        # broken if not instantiated with a PHCsystem
        if self.system:
            self.system.polish()
            self.extract_info()

    def Tillmann_points(self):
        # broken if not instantiated with a PHCsystem
        if self.system is None:
            return []
        result = []
        for n, s in enumerate(self.solutions):
            if (s.t != 1.0 or self.shapes[n].is_degenerate()):
                result.append(n)
        return result

    def permutation(self, other):
        """
        return a list of pairs (m, n) where self.shapes[m] is
        closest to other.shapes[n].
        """
        result = []
        target = set(range(len(other.shapes)))
        for m, shape in enumerate(self.shapes):
            dist, n = min([(shape.dist(other.shapes[k]), k) for k in target])
            result.append( (m, n) )
            target.remove(n)
        return result

    def PHCtransport(self, target_holonomy, allow_collisions=False):
        """
        Use PHC to transport this fiber to a different target holonomy.
        Can only be used if this fiber has a PHCSystem.
        """
        # Not used.
        target_system = self.parametrized_system.transport(
            self.system, target_holonomy, allow_collisions)
        return Fiber(self.hp_manifold, target_holonomy, PHCsystem=self.system,
                     gluing_system=self.gluing_system)

    def transport(self, target_holonomy, allow_collisions=False, debug=False):
        """
        Transport this fiber to a different target holonomy.
        """
        shapes = []
        dT = 1.0
        while True:
            if dT < 1.0/64:
                raise ValueError('Collision unavoidable. Try a different radius.')
            for shape in self.shapes:
                Zn = self.gluing_system.track(shape.array,
                                              target_holonomy,
                                              dT=dT,
                                              debug=debug)
                shapes.append(Zn)
            result = Fiber(self.hp_manifold, target_holonomy,
                           gluing_system=self.gluing_system,
                           shapes=shapes)
            if result.collision():
                dT *= 0.5
            else:
                break
        return result

    def polished_shape(self, n, target_holonomy=None, dec_prec=None, bits_prec=200):
        if target_holonomy is None:
            target_holonomy = self.H_meridian
        return PolishedShapes(self[n], target_holonomy,
                             dec_prec=dec_prec, bits_prec=bits_prec)

class Fibrator:
    """
    A factory for Fibers, used to construct an initial Fiber.  Either loads
    a pre-computed Fiber from a file, or uses PHC to construct one.
    """
    def __init__(self, manifold, target=None, fiber_file=None, tolerance=1.0E-5):
        # The tolerance is used to decode when PHC solutions are regarded
        # as being at infinity.
        if target is None and fiber_file is None:
            raise ValueError('Supply either a target or a saved base fiber.')
        self.manifold = manifold
        self.manifold_name = manifold.name()
        self.target = target
        self.fiber_file = fiber_file
        self.tolerance=tolerance

    def __call__(self):
        """Construct a Fiber, or read in a precomputed Fiber, and return it."""
        fiber_file = self.fiber_file
        signature = self.manifold._to_bytes()
        if fiber_file and os.path.exists(fiber_file):
            print 'Loading the starting fiber from %s'%fiber_file
            with open(fiber_file) as datafile:
                data = eval(datafile.read())
            assert data['signature'] == signature, 'Triangulations do not match!'
            return data['fiber']
        else:
            print 'Computing the starting fiber ... ',
            begin = time.time()
            N = self.manifold.num_tetrahedra()
            variables = ( ['X%s'%n for n in range(N)] +
                      ['Y%s'%n for n in range(N)] )
            self.ring = PolyRing(variables + ['t'])
            self.equations = self.build_equations()
            self.equations += ['X%s + Y%s - 1'%(n,n) for n in range(N)] 
            self.parametrized_system = ParametrizedSystem(
                self.ring,
                't',
                [PHCPoly(self.ring, e) for e in self.equations]
            )
            self.base_system = self.parametrized_system.start(
                self.target, self.tolerance)
            result = Fiber(self.manifold, self.target,
                           PHCsystem=self.base_system)
            print 'done. (%.3f seconds)'%(time.time() - begin)
            if fiber_file:
                with open(fiber_file, 'w') as datafile:
                    datafile.write("{\n'fiber': %s,\n'signature': %s\n}"%(
                        repr(result),
                        repr(signature))
                    )
                print 'Saved base fiber as %s'%fiber_file
            return result
                
    def __len__(self):
        return len(self.base_fiber.solutions)
    
    def rect_to_PHC(self, eqn, rhs=None):
        A, B, c = eqn
        left = []
        if rhs is None:
            right = []
        elif isinstance(rhs, str):
            right = [rhs]
        else:
            right = [str(complex(rhs)).replace('j','*i')]
        for n, a in enumerate(A):
            if a > 0:
                left += ['X%s'%n]*a
            else:
                right += ['X%s'%n]*(-a)
        for n, b in enumerate(B):
            if b > 0:
                left += ['Y%s'%n]*b
            else:
                right += ['Y%s'%n]*(-b)
        if len(left) == 0:
            left = ['1']
        if len(right) == 0:
            right = ['1']
        op = ' - ' if c == 1 else ' + '
        return '*'.join(left) + op + '*'.join(right)

    def build_equations(self):
        if (self.manifold.num_cusps() != 1 or
            not self.manifold.is_orientable()):
            raise ValueError('Manifold must be orientable with one cusp.')
        eqns = self.manifold.gluing_equations('rect')
        meridian = eqns[-2]
        result = []
        for eqn in eqns[:-3]:
            result.append(self.rect_to_PHC(eqn))
        result.append(self.rect_to_PHC(meridian, rhs='t'))
        return result

    def PHC_to_complex(self, line):
       var, colon, real, imag = line.split()
       index=int(var[1:])
       if imag[0] == '-':
	   op = ''
       else:
	   op = '+'
       try:
	   return var[0], index, complex('%s%s%sj'%(real,op,imag))
       except ValueError:
	   print 'PHC parse error on %s+%sj'%(real,imag)

class CircleElevation:
    """A family of fibers for the meridian holonomy map, lying above the
    points Rξ^m where ξ = e^(2πi/N). The value of N is specified by
    the keyword argument *order* (default 128) and the value of R is
    specified by the keyword argument *radius* (default 1.02).

    The construction begins by using PHC to find a single fiber over
    the point ξ_0 = Re^(2πi/N_0).  The value of N_0 can be specified
    with the keyword argument *base_index*.  The default behavior is
    to choose N_0 at random.

    Once a base Fiber has been constructed, it is transported around
    the circle, using Newton's method, to construct the full family of
    Fibers.

    A CircleElevation can be tightened, which means radially
    transporting each fiber lying over a point on the R-circle to one
    lying over a point on the circle of radius T. (T = 1.0 by
    default.) Singularities are common on the unit circle, and may
    prevent the transport. Such failures are reported on the console
    and then ignored.

    """
    def __init__(self, manifold, order=128, radius=1.02, target_arg=None,
                 base_fiber_file=None):
        self.order = order
        self.radius = radius
        self.manifold = manifold
        self.hp_manifold = self.manifold.high_precision()
        self.betti2 = [c % 2 for c in manifold.homology().coefficients].count(0)
        Darg = 2*pi/order
        # The minus sign is for consistency with the sign convention of numpy.fft
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
        self.fibrator = Fibrator(manifold, target=target, fiber_file=base_fiber_file)
        self.base_fiber = base_fiber = self.fibrator()
        arg = log(base_fiber.H_meridian).imag%(2*pi)
        self.base_index = (self.order - int(arg*self.order/(2*pi)))%self.order
        if not base_fiber.is_finite():
            raise RuntimeError, 'The starting fiber contains Tillmann points.'
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
        self.M_holo, self.L_holo = [Glunomial(A,B,c) for A,B,c in eqns[-2:]]
        self.glunomials.append(self.M_holo)
        self.track_satellite()
        self.R_longitude_holos, self.R_longitude_evs = self.longidata(
            self.R_fibers)
            
    def __call__(self, Z):
        return array([F(Z) for F in self.glunomials])

    def track_satellite(self):
        """
        Construct the fibers over the circle of radius R.
        """
        print 'Tracking the satellite at radius %s ...'%self.radius
        start = time.time()
        arg = log(self.base_fiber.H_meridian).imag%(2*pi)
        R = self.radius
        circle = self.R_circle
        base_index = self.base_index
        print 'Base index is %s'%base_index
        print ' %-5s\r'%base_index,
        # Move to the R-circle, if necessary.
        self.R_fibers[base_index] = self.base_fiber.transport(circle[base_index])
        for n in xrange(base_index+1, self.order):
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
        for n in xrange(base_index-1, -1, -1):
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

    def tighten(self, T=1.0):
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
        print '\nChecking for Tillmann points.'
        for n in xrange(self.order):
            try:
                t = self.T_fibers[n].Tillmann_points()
                if t:
                    print 'Tillmann points %s found in fiber %s.'%(t, n)
            except AttributeError: # If the fiber was not computed.
                print ' Skipping %s'%n
        self.T_longitude_holos, self.T_longitude_evs = self.longidata(self.T_fibers)

    def longidata(self, fiber_list):
        """Compute the longitude holonomies and eigenvalues at each point in
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
            [( n, self.L_holo(f.shapes[m].array) ) for n, f in enumerate(fiber_list)
             if isinstance(f, Fiber)]
            for m in xrange(self.degree)]
        if isinstance(fiber_list[0], Fiber):
            index = 0;
        else:
            index = randint(0,self.order - 1)
            print 'Using %d as the starting index.'%index 
        longitude_traces = self.find_longitude_traces(fiber_list[index])
        longitude_eigenvalues = []
        for m, L in enumerate(longitude_holonomies):
            tr = longitude_traces[m]
            n, holo = L[index]
            e = sqrt(holo)
            # Choose the sign for the eigenvalue at the starting fiber
            E = [ (n,e) if abs(e + 1/e - tr) < abs(e + 1/e + tr) else (n,-e) ]
            # Avoid discontinuities caused by the branch cut used by sqrt
            for n, holo in L[index+1:]:
                e = sqrt(holo)
                E.append( (n,e) if abs(e - E[-1][1]) < abs(e + E[-1][1]) else (n,-e) )
            if index > 0:
                for n, holo in L[index-1::-1]:
                    e = sqrt(holo)
                    E.insert(0,(n,e) if abs(e - E[0][1]) < abs(e + E[0][1]) else (n,-e) )
            longitude_eigenvalues.append(E)
        return longitude_holonomies, longitude_eigenvalues

    def permutation(self, fiber_list):
        result = Permutation()
        start, end = fiber_list[0].shapes, fiber_list[-1].shapes
        for n, shape in enumerate(start):
            distances = array([shape.dist(end_shape) for end_shape in end])
            result[n] = argmin(distances)
        return result

    def compute_volumes(self, fiber_list):
        volumes = [ [] for n in range(self.degree) ]
        for fiber in fiber_list:
            for n, shape in enumerate(fiber.shapes):
                self.manifold.set_tetrahedra_shapes(shape(), fillings=[(0,0)])
                volumes[n].append(self.manifold.volume())
        return volumes
        
    def find_longitude_traces(self, fiber):
        # Sage complex numbers do not support attributes .real and .imag :^(((
        trace = lambda rep : complex(rep[0,0] + rep[1,1])
        traces = []
        for shape in fiber.shapes:
            a = shape.array
            self.hp_manifold.set_tetrahedra_shapes(a, a, [(0,0)])
            G = self.hp_manifold.fundamental_group()
            longitude = G.peripheral_curves()[0][1]
            relators = G.relators()
            generators = G.generators()
            M, N = len(relators), G.num_generators()
            A = matrix(zeros((M,N),'i'))
            L = zeros(N,'i')
            rhs = zeros(M,'i')
            for i in range(M):
                for j in range(N):
                    A[i,j] = (relators[i].count(generators[j]) +
                               relators[i].count(generators[j].upper()))%2
                    L[j] = (longitude.count(generators[j]) +
                               longitude.count(generators[j].upper()))%2
                rhs[i] = trace(G.SL2C(relators[i])).real < 0
            S = matrix(solve_mod2_system(A,rhs)).transpose()
            # Paranoia
            if max((A*S - matrix(rhs).transpose())%2) > 0:
                if self.betti2 == 1:
                    raise RuntimeError, "Mod 2 solver failed!"
            tr = trace(G.SL2C(longitude))
            if (L*S)%2 != 0:
                tr = -tr
            traces.append(tr)
        return traces

    def show_R_longitude_evs(self):
        R_plot = Plot([[x for n,x in track] for track in self.R_longitude_evs])

    def show_T_longitude_evs(self):
        T_plot = Plot([[x for n,x in track] for track in self.T_longitude_evs])

    def holo_permutation(self):
        return [self.R_fibers[0].shapes.index(p) for p in self.last_R_fiber.shapes]

    def holo_orbits(self):
        P = self.holo_permutation()
        Q = list(P)
        orbits = []
        while Q:
            first_one = this_one = Q.pop()
            orbit = []
            while True:
                orbit.append(this_one)
                this_one = P[this_one]
                if this_one == first_one:
                    break
                else:
                    Q.remove(this_one)
            orbits.append(orbit)
        return orbits

def solve_mod2_system(the_matrix,rhs):
    M,N = the_matrix.shape
    A = zeros((M,N+1),'i')
    A[:,:-1] = the_matrix
    A[:,-1] = rhs
    S = zeros(N,'i')
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
        P.insert(0,j)
        for i in range(r+1,M):
            if A[R[i]][j] == 1:
                A[R[i]] = A[R[i]]^A[R[r]]
        r += 1
    i = len(P)-1 
    for j in P:
        S[j] = (A[R[i]][N] - dot(A[R[i]][j+1:-1], S[j+1:]))%2
        i -= 1
    return S


class PEArc(list):
    """
    A list of pillowcase points lying on an arc of the PECharVariety.
    Subclassed here to allow additional attributes.
    """

class PECharVariety:
    def __init__(self, manifold, order=128, radius=1.02,
                 elevation=None, base_dir='PE_base_fibers', hint_dir='hints'):
        if isinstance(manifold, Manifold):
            self.manifold = manifold
        else:
            self.manifold = Manifold(manifold)
        self.radius = radius
        self.order = order
        self.hint_dir = hint_dir
        if elevation is None:
            self.check_dir(base_dir, 'I need a directory for storing base fibers.')
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

    def check_dir(self, dir, message=''):
        if not os.path.exists(dir):
            cwd = os.path.abspath(os.path.curdir)
            newdir = os.path.join(cwd,dir)
            print '\n'+ message
            response = raw_input("May I create a directory %s?(Y/n)"%newdir)
            if response and response.lower()[0] != 'y':
                sys.exit(0)
            print
            os.mkdir(newdir)

    def save_hint(self, basename=None, dir=None):
        if dir == None:
            dir = self.hint_dir
        self.check_dir(dir, 'I need a directory for storing hint files.')
        if basename == None:
            basename = self.manifold.name()
        hintfile_name = os.path.join(dir, basename + '.hint')
        hintfile = open(hintfile_name,'w')
        hintfile.write('hint={\n')
        hintfile.write('"manifold" : %s,\n'%self.manifold)
        hintfile.write('"radius" : %f,\n'%self.radius)
        hintfile.write('"order" : %d,\n'%self.order)
        hintfile.write('}\n')
        hintfile.close()

    def build_arcs(self, show_group=False):
        self.arcs = []
        self.arc_info = []
        H = self.elevation
        delta_M = -1.0/self.order
        M_args = 0.5 * ( arange(self.order, dtype=float64)*delta_M % 1.0 )
        for m, track in enumerate(self.elevation.T_longitude_evs):
            arc, info = PEArc(), []
            marker = ''
            for n, ev in track:
                if (0.99999 < abs(ev) < 1.00001):
                    if show_group:
                        shape = H.T_fibers[n].shapes[m]
                        try:
                            if shape.in_SU2():
                                marker = '.'
                            elif shape.has_real_traces():
                                marker = 'D'
                            else:
                                marker = 'x'
                        except:
                            print 'Exception in group test.'
                            marker = 'x'
                    L = (log(ev).imag/(2*pi))%1.0
                    if len(arc)>2:  # don't do this near the corners.
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
                    arc.append(PEPoint(L,M_args[n], marker=marker))
                    info.append( (m,n) )
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
                    if arc[0].imag > 0.45 and arc[1].imag  < 0.05 :
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
        arcs = list(self.arcs)
        # caps
        arcs.sort(key=lambda x : x[0].imag, reverse=True)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[0].imag == arcs[0][0].imag:
                    level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].first_shape.dist(a.first_shape)
                             for a in level[1:]])
                cap = [level.pop(0), level.pop(distances.argmin())]
                cap.sort(key=lambda a : a[0].real)
                left, right = cap
                if .05 < right[0].imag < .45:
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
        arcs.sort(key=lambda x : x[-1].imag)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[-1].imag == arcs[0][-1].imag:
                    level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].last_shape.dist(a.last_shape)
                             for a in level[1:]])
                cup = [level.pop(0), level.pop(distances.argmin())]
                cup.sort(key=lambda a : a[-1].real)
                left, right = cup
                if 0.05 < right[-1].imag < 0.45:
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
        self.build_arcs(show_group)
        term = 'aqua' if sys.platform == 'darwin' else 'wxt'
        Plot(self.arcs,
             limits=((0.0, 1.0), (0.0, 0.5)),
             margins=(0,0),
             aspect='equal',
             title=self.manifold.name(),
             colors = self.colors,
             extra_lines=[((0.5,0.5),(0.0,1.0))],
             extra_line_args={'color':'black', 'linewidth':0.75},
             show_group=show_group,
             )

class Permutation(dict):
    def orbits(self):
        points = set(self.keys())
        orbits = []
        while points:
            first = n = points.pop()
            orbit = [first]
            while True:
                n = self[n]
                if n == first:
                    orbits.append(orbit)
                    break
                else:
                    points.remove(n)
                    orbit.append(n)
        return orbits
