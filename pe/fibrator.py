import os, time
from gluing import GluingSystem
try:
    from phc import PolyRing, PHCPoly, PHCSystem, ParametrizedSystem
except ImportError:
    print('No phc module, so will only work with precomputed fibers')
from fiber import Fiber
from snappy import Manifold, ManifoldHP

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
