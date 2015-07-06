import numpy
from gluing import GluingSystem
from shape import Shapes

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
