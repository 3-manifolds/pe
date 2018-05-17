"""
Define the Fibrator class.

A Fibrator object uses PHC to construct an initial fiber, which can
then be transported around a circle.
"""
from __future__ import print_function
import os, time
try:
    from cyphc import PolyRing, PHCPoly, ParametrizedSystem
except ImportError:
    print('No phc module, so will only work with precomputed fibers')
from .fiber import Fiber

class Fibrator(object):
    """
    A factory for Fibers, used to construct an initial Fiber.  Either loads
    a pre-computed Fiber from a file, or uses PHC to construct one.
    """
    def __init__(self, manifold, target=None, shapes=None, tolerance=1.0E-6, base_dir=None):
        # The tolerance is used to decode when PHC solutions are regarded
        # as being at infinity.
        self.base_dir = base_dir
        if target is None and shapes is None:
            raise ValueError('Supply either a target or a list of shapes.')
        self.manifold = manifold
        self.manifold_name = manifold.name()
        self.target = target
        self.shapes = shapes
        self.tolerance = tolerance

    def __call__(self):
        """
        Return a base Fiber constructed from scratch or from precomputed shapes."""
        if self.shapes:
            return Fiber(self.manifold, self.target, shapes=self.shapes)
        else:
            print('Computing the base fiber ... ', end=' ')
            begin = time.time()
            result = self.PHC_compute_fiber(self.target)
            print('done. (%.3f seconds)'%(time.time() - begin))
            if self.base_dir:
                base_fiber_file=os.path.join(self.base_dir, self.manifold.name()+'.base')
                template="{{\n'manifold': '''{mfld}''',\n'H_meridian': {target},\n'shapes': {shapes}\n}}"
                shape_repr = repr([list(s) for s in result.shapes])
                shape_repr = shape_repr.replace(',', ',\n').replace('[[','[\n [')
                with open(base_fiber_file, 'w') as datafile:
                    datafile.write(template.format(
                        mfld=self.manifold._to_string(),
                        target=self.target,
                        shapes=shape_repr))
                print('Saved base fiber as %s'%base_fiber_file)
            return result

    def PHC_compute_fiber(self, target, tolerance=None):
        if tolerance is None:
            tolerance = self.tolerance
        target = complex(target) # in case we were passed a Sage number
        N = self.manifold.num_tetrahedra()
        variables = (['X%s'%n for n in range(N)] + ['Y%s'%n for n in range(N)])
        ring = PolyRing(variables + ['t'])
        equations = self.build_equations()
        equations += ['X%s + Y%s - 1'%(n, n) for n in range(N)]
        parametrized_system = ParametrizedSystem(ring, 't',
                                                 [PHCPoly(ring, e) for e in equations])
        base_system = parametrized_system.start(target, tolerance)
        return Fiber(self.manifold, target, PHCsystem=base_system)

    @staticmethod
    def rect_to_PHC(eqn, rhs=None):
        """Convert a system of gluing equations to PHC's format."""
        A, B, c = eqn
        left = []
        if rhs is None:
            right = []
        elif isinstance(rhs, str):
            right = [rhs]
        else:
            right = [str(complex(rhs)).replace('j', '*i')]
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
        """Use SnapPy to construct a system of gluing equations."""
        if self.manifold.num_cusps() != 1 or not self.manifold.is_orientable():
            raise ValueError('Manifold must be orientable with one cusp.')
        eqns = self.manifold.gluing_equations('rect')
        meridian = eqns[-2]
        result = []
        for eqn in eqns[:-3]:
            result.append(self.rect_to_PHC(eqn))
        result.append(self.rect_to_PHC(meridian, rhs='t'))
        return result

    def write_phc_file(self, filename):
        """
        Save the system of equations defining the entire gluing variety
        in PHC format.
        """
        if self.manifold.num_cusps() != 1 or not self.manifold.is_orientable():
            raise ValueError('Manifold must be orientable with one cusp.')
        N = self.manifold.num_tetrahedra()
        eqns = self.manifold.gluing_equations('rect')
        system = []
        for eqn in eqns[:-3]:
            system.append(self.rect_to_PHC(eqn))
        system += ['X%s + Y%s - 1'%(n, n) for n in range(N)]
        with open(filename, 'wb') as output:
            output.write('%d %d\n'%(2*N - 1, 2*N))
            for equation in system:
                output.write(equation + ';\n')

