"""
Define the GluingSystem and Glunomial classes.

A GluingSystem object represents a system of gluing equations for an
ideal triangulation of a 3-manifold.

The monomials in the equations are represented by Glunomial objects.
"""

from __future__ import print_function
from numpy import dtype, ndarray, array, matrix, prod, ones, pi, exp
from numpy.linalg import svd, norm, solve, lstsq, matrix_rank
from numpy.random import random
from snappy.snap.shapes import enough_gluing_equations

# Constants for Newton's method
RESIDUAL_BOUND = 1.0E-14
STEPSIZE_BOUND = 1.0E-15

class Glunomial(object):
    """
    A product of powers of linear terms z_i or (1-z_i), as appears on
    the left side of a gluing equation.  These are Laurent monomials,
    so powers may be negative.

    Instantiate a glunomial with a triple, as in the list returned
    by Manifold.gluing_equations('rect').
    """
    def __init__(self, A, B, c):
        self.A, self.B, self.sign = A, B, c

    def __repr__(self):
        apower = lambda n, p: 'z%d^%s'%(n, p) if p != 1 else 'z%s'%n
        bpower = lambda n, p: '(1-z%d)^%s'%(n, p) if p != 1 else '(1-z%s)'%n
        Apowers = [apower(n, a) for n, a in enumerate(self.A) if a != 0]
        Bpowers = [bpower(n, b) for n, b in enumerate(self.B) if b != 0]
        sign = '' if self.sign == 1.0 else '-'
        return sign + '*'.join(Apowers+Bpowers)

    def __call__(self, Z):
        """
        Evaluate this monomial on a numpy array of shapes.  The shapes may
        be numpy complex128 numbers, or elements of a Sage ComplexField,
        or mpmath complex numbers.
        """
        assert isinstance(Z, ndarray)
        W = 1 - Z
        try:
            return self.sign*prod(Z**self.A)*prod(W**self.B)
        except ValueError:
            print('Glunomial evaluation failed on %s'%self)
            print('A =', self.A)
            print('B =', self.B)
            print('c =', self.sign)
            print('Z =', Z, '(%s)'%type(Z[0]))
            raise ValueError

    def gradient(self, Z):
        """
        Return the gradient of this monomial evaluated at a numpy array
        of shapes.
        """
        assert isinstance(Z, ndarray)
        W = 1 - Z
        return self.sign*prod(Z**self.A)*prod(W**self.B)*(self.A/Z - self.B/W)

class GluingSystem(object):
    """
    The system of gluing equations for an ideal triangulaton of a
    one-cusped 3-manifold.
    """

    def __init__(self, manifold):
        assert manifold.num_cusps() == 1, 'Manifold must be one-cusped.'
        self.manifold = manifold
        self.num_shapes = manifold.num_tetrahedra()
        eqns = enough_gluing_equations(manifold)
        self.glunomials = [Glunomial(A, B, c) for A, B, c in eqns]
        cusp_eqns = manifold.gluing_equations('rect')[-2:]
        self.M_nomial, self.L_nomial = [Glunomial(A, B, c) for A, B, c in cusp_eqns]

    def __repr__(self):
        return '\n'.join([str(G) for G in self.glunomials])

    def __call__(self, Z):
        return array([G(Z) for G in self.glunomials])

    def __len__(self):
        return len(self.glunomials)

    def jacobian(self, Z):
        """Return the Jacobian matrix for the system at a point Z in shape space."""
        return matrix([G.gradient(Z) for G in self.glunomials])

    def M_holonomy(self, Z):
        """Evaluate the holonomy function of the meridian at a point Z in shape space."""
        return complex(self.M_nomial(Z))

    def L_holonomy(self, Z):
        """Evaluate the holonomy function of the longitude at a point Z in shape space."""
        return complex(self.L_nomial(Z))

    def corank(self, Z):
        """
        Return the corank of the Jacobians at the point Z for the
        defining equations of the gluing variety.  Raise an assertion
        error if the augmented system, with the equation H_M = target
        added, does not have corank one less.
        """
        jacobian = self.jacobian(Z)
        gluing_rank = matrix_rank(jacobian[:-1])
        system_rank = matrix_rank(jacobian)
        if gluing_rank != system_rank - 1:
            print('gluing rank:', gluing_rank, 'system rank:', system_rank,
                  'dimension:', self.num_shapes - system_rank)
            jacobian[-1] = self.L_nomial.gradient(Z)
            print('G + long rank:', matrix_rank(jacobian))
            return -1
        return self.num_shapes - system_rank

    def newton_error(self, Z):
        """
        This is meant to bound the change in the Meridian holonomy within
        a ball of radius STEPSIZE_BOUND around the shape vector Z.
        """
        return self.num_shapes*norm(self.M_nomial.gradient(Z), 1)*STEPSIZE_BOUND

    def newton_step(self, Z, M_target):
        """
        Do one iteration of Newton's method, starting at Z and aiming
        to solve G(z) = (1,1,...,M_target).  Returns a triple:
        Z', step_size, residual.  Solves the linear system by
        LU factorization (not great for nearly singular systems).
        """
        J = self.jacobian(Z)
        target = ones(len(self), dtype='complex128')
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
        target = ones(len(self), dtype='complex128')
        target[-1] = M_target
        error = target - self(Z)
        dZ = lstsq(J, error, rcond=-1)[0]
        return dZ, target

    def newton1(self, Z, M_target, debug=False):
        """
        Simple version of Newton's method.  Uses the LU decomposition to
        solve the linear system.  Does not adjust step sizes.
        The iteration is terminated if:
          * the residual does not decrease; or
          * the step size is smaller than STEPSIZE_BOUND
          * more than 10 iterations have been attempted
        """
        prev_residual = step_size = 1.0E5
        prev_Z, count, res_bound = Z, 1, self.newton_error(Z)
        while True:
            Zn, step_size, residual = self.newton_step(prev_Z, M_target)
            if debug:
                print(count, residual, step_size)
            if residual > prev_residual:
                return prev_Z, prev_residual
            if step_size < STEPSIZE_BOUND or residual < res_bound or count > 10:
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
        prev_Z, count, res_bound = Z, 1, self.newton_error(Z)
        while True:
            dZ, target = self.newton_step_ls(prev_Z, M_target)
            t = 1.0
            for _ in range(5):
                Zn = prev_Z + t*dZ
                residual = max(abs(target - self(Zn)))
                if residual < prev_residual:
                    break
                else:
                    t *= 0.5
                    if debug:
                        print('reducing t to %s'%t)
            if debug:
                print('scaled dZ by %s; residual: %s'%(t, residual))
            if residual > prev_residual:
                if debug:
                    print('Armijo failed with t=%s'%t)
                return prev_Z, prev_residual
            step_size = norm(t*dZ)
            if step_size < STEPSIZE_BOUND or residual < res_bound or count > 10:
                return Zn, residual
            prev_Z, prev_residual = Zn, residual
            count += 1

    def track(self, Z, M_target, dT=1.0, debug=False, fail_quietly=False):
        """
        Track solutions of the gluing system starting at Z and
        ending at a solution where the meridian holonomy takes the
        specified value.  The path is subdivided into subpaths of
        length determined by dT, which may be further divided if
        convergence problems arise.  Returns a shape array and a
        boolean indicating success.
        """
        M_start = self(Z)[-1]
        delta = (M_target - M_start)
        T = 0.0
        Zn = Z
        if debug:
            print('tracking to target ', M_target)
            print('Corank:', self.corank(Z))
        # First we try the cheap and easy method
        target = M_start + delta
        Zn, residual = self.newton1(Zn, target)
        if residual < 1.0E-8: # What is a good threshold here?  Was 1.0E-12
            return Zn, ''
        # If that fails, try taking baby steps.
        if debug:
            print('Taking baby steps ...')
        success = 0
        T, dT = 0.0, 0.5*dT
        prev_Z = Z
        while T < 1.0:
            Tn = min(T+dT, 1.0)
            if debug:
                print('trying T = %.17f'%Tn)
            baby_target = M_start + Tn*delta
            Zn, residual = self.newton2(prev_Z, baby_target, debug=debug)
            if residual < 1.0E-8: # was 1.0E-12
                # After 3 successful baby steps, double the step size.
                prev_Z = Zn
                if success > 3:
                    success = 0
                    dT *= 2
                    if debug:
                        print('Track step increased to %.17f'%dT)
                else:
                    success += 1
                T = Tn
            else:
                # If we fail, halve the step size and try again.  Give up after 16 failures.
                success = 0
                dT /= 2
                if debug:
                    print('Track step reduced to %.17f; corank = %s'%(dT, self.corank(prev_Z)))
                if dT < STEPSIZE_BOUND:
                    if fail_quietly:
                        return Zn, 'Track failed: step size limit reached.'
                    else:
                        print('\nLongitude holonomy:', self.L_holonomy(Zn))
                        print('Track parameter:', Tn)
                        print('Shapes:', Zn)
                        print('Corank:', self.corank(Z))
                        print('Newton bound:', self.newton_error(prev_Z))
                        print('residual:', residual)
                        raise ValueError('Track failed: step size limit reached.')
        return Zn, ''
