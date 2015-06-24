"""
Using PHC to find solutions to the gluing equations for closed manifolds.
"""

import sys
import snappy
from .polish_reps import PSL2CRepOf3ManifoldGroup, CheckRepresentationFailed
from .real_reps import PSL2RRepOf3ManifoldGroup, CouldNotConjugateIntoPSL2R
from . import phc_hack

def clean_complex(z, epsilon=1e-20):
    r, i = abs(z.real), abs(z.imag)
    if r < epsilon and i < epsilon:
        ans = 0.0
    elif r < epsilon:
        ans = z.imag*1j
    elif i < epsilon:
        ans = z.real
    else:
        ans = z
    assert abs(z - ans) < epsilon
    return ans

class PHCGluingSolutionsOfClosed:
    """
    >>> M = snappy.Manifold('m004(1,2)')
    >>> ans = PHCGluingSolutionsOfClosed(M).solutions()
    >>> map(len, ans)
    [2, 0, 8, 4]
    """
    def __init__(self, manifold):
        if isinstance(manifold, str):
            manifold = snappy.Manifold(manifold)
        else:
            manifold = manifold.copy()
        if True in manifold.cusp_info('complete?') or not manifold.is_orientable():
            raise ValueError("Manifold must be closed and orientable")

        manifold.set_peripheral_curves('fillings')
        self.manifold = manifold
        self.N = N = manifold.num_tetrahedra()
        self.variables = ['X%s'%n for n in range(N)] + ['Y%s'%n for n in range(N)]
        self.equations = [self.rect_to_PHC(eqn) for eqn in
                          snappy.snap.shapes.enough_gluing_equations(manifold)]
        self.equations += ['X%s + Y%s - 1'%(n,n) for n in range(N)]
        self.phc_system = None
        
        
    def raw_solutions(self, max_err=1e-6):
        import phc
        
        if self.phc_system is None:
            self.ring = phc.PolyRing(self.variables)        
            self.system = phc.PHCSystem(self.ring,
                                    [phc.PHCPoly(self.ring, eqn) for eqn in self.equations])
        ans = []
        try:
            sols = self.system.solution_list()
        except phc.PHCInternalAdaException:
            return []
        for sol in sols:
            if sol.err < max_err:
                ans.append([clean_complex(z) for z in sol.point[:self.N]])
        return ans

    def solutions(self, working_prec=230):
        psl2Rtilde, psl2R, su2, rest = [], [], [], []
        for sol in self.raw_solutions():
            rho = PSL2CRepOf3ManifoldGroup(self.manifold,
                            target_meridian_holonomy_arg=0,
                            rough_shapes=sol)
            try:
                rho.polished_holonomy(working_prec)
            except CheckRepresentationFailed:
                continue
            if rho.appears_to_be_SU2_rep():
                su2.append(sol)
            elif rho.is_PSL2R_rep():
                try:
                    rho = PSL2RRepOf3ManifoldGroup(rho)
                    if rho.representation_lifts():
                        psl2Rtilde.append(sol)
                    else:
                        psl2R.append(sol)
                except (CouldNotConjugateIntoPSL2R, CheckRepresentationFailed):
                    rest.append(sol)
            else:
                rest.append(sol)

        return psl2Rtilde, psl2R, su2, rest
                
    def rect_to_PHC(self, eqn):
        A, B, c = eqn
        left, right = ['1'], ['1']
        for n, a in enumerate(A):
            if a > 0:
                left += ['X%s^%s'% (n, a)]
            elif a < 0:
                right += ['X%s^%s'% (n, -a)]
        for n, b in enumerate(B):
            if b > 0:
                left += ['Y%s^%s'% (n, b)]
            elif b < 0:
                right += ['Y%s^%s'% (n, -b)]
        op = ' - ' if c == 1 else ' + '
        return '*'.join(left) + op + '*'.join(right)

class PHCGluingSolutionsOfClosedStandalone(
        PHCGluingSolutionsOfClosed):
    """
    Uses command line PHC, connected by pipes, rather than
    the CyPHC library.

    Note: Uses PHC's blackbox solver mode, which is sometimes
    inferior to the parameter choices made by CyPHC.

    >>> M = snappy.Manifold('m004(1,2)')
    >>> ans = PHCGluingSolutionsOfClosedStandalone(M).solutions()
    >>> map(len, ans)
    [2, 0, 8, 4]
    """
    def raw_solutions(self):
        import sage.interfaces.phc as sage_phc
        from sage.all import PolynomialRing, QQ
        
        vars = self.variables
        R = PolynomialRing(QQ, vars)
        x_vars = [R(x) for x in vars[:len(vars)/2]]
        sols = sage_phc.phc.blackbox(map(R, self.equations), R)
        sols_dicts = sage_phc.get_classified_solution_dicts(
            sols.output_file_contents, R)
        good_sols = sols_dicts['real'] + sols_dicts['complex']
        ans = [ [clean_complex(complex(sol[x])) for x in x_vars]
                for sol in good_sols]
        return ans

class PHCGluingSolutionsOfClosedHack(
        PHCGluingSolutionsOfClosed):
    """
    To avoid memory leaks and random PARI crashes, runs CyPHC
    in a separate subprocess.

    >>> M = snappy.Manifold('m004(1,2)')
    >>> ans = PHCGluingSolutionsOfClosedHack(M).solutions()
    >>> map(len, ans)
    [2, 0, 8, 4]
    """
    def raw_solutions(self):
        import subprocess
        args = [sys.executable, phc_hack.__file__,
                ','.join(self.variables)] + self.equations
        P = subprocess.Popen(args, stdout=subprocess.PIPE)
        try:
            ans = eval(P.stdout.read())
        except:
            ans = []
        return ans
    

if __name__=='__main__':
    import doctest
    doctest.testmod()
