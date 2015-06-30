import snappy
from snappy.snap.shapes import (
    float_to_pari, complex_to_pari, pari_column_vector, prec_bits_to_dec,
    infinity_norm, pari_matrix, pari_vector_to_list,
    enough_gluing_equations, eval_gluing_equation, _within_sage,
)
from numpy import array, matrix
from numpy.linalg import norm

if _within_sage:
    from sage.all import exp, CC, ComplexField, pari, gen
else:
    from Cypari import pari, gen

def pari_set_precision(x, dec_prec):
    return pari(0) if x == 0 else pari(x).precision(dec_prec)

class GoodShapesNotFound(Exception):
    pass

class Shape(object):
    """
    A vector of shape parameters, stored as a numpy.array.
    Many methods are available: ...
    Instantiate with a sequence of complex numbers.
    """
    def __init__(self, manifold, values, tolerance=1.0E-6):
        if isinstance(manifold, snappy.ManifoldHP):
            self.hp_manifold = manifold
        else:
            self.hp_manifold = manifold.high_precision()
        self.array = array(values)

    def __str__(self):
        return self.array.__str__()

    def __getitem__(self, value):
        return self.array[value]

    def __len__(self):
        return len(self.array)
        
    def __eq__(self, other):
        return norm(self.array - other.array) < 1.0E-6

    def dist(self, other):
        return norm(self.array - other.array)

    def __repr__(self):
        return repr(list(self))
    
    def is_degenerate(self):
        moduli = abs(self.array)
        return ( (moduli < 1.0E-6).any() or
                 (moduli < 1.0E-6).any() or
                 (moduli > 1.0E6).any()
                 )

    def SL2C(self, word):
        self.hp_manifold.set_tetrahedra_shapes(self.array, None, [(0,0)])
        G = self.hp_manifold.fundamental_group()
        return G.SL2C(word)

    def O31(self, word):
        self.hp_manifold.set_tetrahedra_shapes(self.array, None, [(0,0)])
        G = self.hp_manifold.fundamental_group()
        return G.O31(word)

    def has_real_traces(self):
        tolerance = 1.0E-10
        gens = self.hp_manifold.fundamental_group().generators()
        gen_mats = [self.SL2C(g) for g in gens]
        for A in gen_mats:
            tr = complex(A[0,0] + A[1,1])
            if abs(tr.imag) > tolerance:
                return False
        mats = gen_mats[:]
        for i in range(1, len(gens) + 1):
            new_mats = []
            for A in gen_mats:
                for B in mats:
                    C = B*A
                    tr = complex(C[0,0] + C[1,1])
                    if abs(tr.imag) > tolerance:
                        return False
                    new_mats.append(C)
        return True
        
    def in_SU2(self):
        tolerance = 1.0E-5
        gens = self.hp_manifold.fundamental_group().generators()
        # Check that all generators have real trace in [-2,2]
        for X in [self.SL2C(g) for g in gens]:
            tr = complex(X[0,0] + X[1,1])
            if abs(tr.imag) > tolerance:
                # print 'trace is not real'
                return False
            if abs(tr.real) > 2.0:
                # print 'trace is not in [-2,2]'
                return False
        # Get O31 matrix generators ...
        o31matrices = [real_array(array(self.O31(g))) for g in gens]
        # take the first two, ...
        A, B = o31matrices[:2]
        # find their axes, ...
        M = matrix(zeros((4,4)))
        u, s, v = svd(A - eye(4))
        vt = transpose(v)
        M[:,[0,1]] = vt[:,[n for n in range(4) if abs(s[n]) < tolerance]]
        u, s, v = svd(B - eye(4))
        vt = transpose(v)
        M[:,[2,3]] = vt[:,[n for n in range(4) if abs(s[n]) < tolerance]]
        # check if the axes cross,
        # and find the fixed point (i.e. Minkwoski line)
        u, s, v = svd(M)
        vt = transpose(v)
        rel = vt[:,[n for n in range(4) if abs(s[n]) < tolerance]]
        if rel.shape != (4,1):
            # print 'linear algebra failure'
            return False
        # We now have two descriptions -- let's average them.
        rel[2] = -rel[2]
        rel[3] = -rel[3]
        fix = M*rel
        # Check if the fixed line is in the light cone.
        if abs(fix[0]) <= norm(fix[1:]):
            # print 'fixed line is not in the light cone'
            return False
        # Check if all of the generators fix the same point.
        for O in o31matrices:
            if norm(O*fix - fix) > tolerance:
                # print 'some generators do not share the fixed point.'
                return False
        return True

class PolishedShape(object):
    """A refined Shape containing an arbitrarily precise solution to the
    gluing equations with a specified target value for the meridian
    holonomy.

    >>> M = snappy.Manifold('m071(0,0)')
    >>> alpha = polished_tetrahedra_shapes(M, 0, bits_prec=500)
    >>> M = snappy.Manifold('m071(7,0)')
    >>> beta = polished_tetrahedra_shapes(M, 2*CC.pi()/7, bits_prec=1000)

    """
    def __init__(self, rough_shape, target_holonomy, tolerance=1.0E-6,
                 dec_prec=None, bits_prec=200, ignore_solution_type=False):
        if dec_prec is None:
            dec_prec = prec_bits_to_dec(bits_prec)
        else:
            bits_prec = prec_dec_to_bits(dec_prec)
        working_prec = dec_prec + 10
        target_espilon = pari_set_precision(10.0, working_prec)**-dec_prec
        det_epsilon = pari_set_precision(10.0, working_prec)**-(dec_prec//10)
        init_shapes = pari_column_vector(
            [complex_to_pari(z, working_prec) for z in rough_shape.array])
        self.manifold = manifold = rough_shape.hp_manifold.copy()
        manifold.dehn_fill( (1, 0) ) 
        init_equations = manifold.gluing_equations('rect')
        target = pari.complex(
            pari_set_precision(target_holonomy.real, dec_prec),
            pari_set_precision(target_holonomy.imag, dec_prec))
        if self._gluing_equation_error(
                init_equations, init_shapes, target) > pari(0.000001):
            raise GoodShapesNotFound('Initial solution not very good')
    
        # Now begin the actual computation
        
        eqns = enough_gluing_equations(manifold)
        assert eqns[-1] == manifold.gluing_equations('rect')[-1]
    
        shapes = init_shapes 
        for i in range(100):
            errors = self._gluing_equation_errors(eqns, shapes, target)
            if infinity_norm(errors) < target_espilon:
                break
            shape_list = pari_vector_to_list(shapes)
            derivative = [ [  eqn[0][i]/z  - eqn[1][i]/(1 - z)
                              for i, z in enumerate(shape_list)]
                           for eqn in eqns]
            derivative[-1] = [ target*x for x in derivative[-1] ]
            derivative = pari_matrix(derivative)
    
            det = derivative.matdet().abs()
            if min(det, 1/det) < det_epsilon:
                break  # Pari might crash
            gauss = derivative.matsolve(pari_column_vector(errors))
            shapes = shapes - gauss
    
        # Check to make sure things worked out ok.
        error = self._gluing_equation_error(init_equations, shapes, target)
        total_change = infinity_norm(init_shapes - shapes)
        if error > 1000*target_espilon or total_change > pari(0.0000001):
            raise GoodShapesNotFound('Failed to find solution')
        self.shapes = pari_vector_to_list(shapes)
        if _within_sage:
            CC = ComplexField(bits_prec)
            self.shapes = [CC(z) for z in self.shapes]

    def _gluing_equation_errors(self, eqns, shapes, RHS_of_last_eqn):
        last = [eval_gluing_equation(eqns[-1], shapes) - RHS_of_last_eqn]
        return [eval_gluing_equation(eqn, shapes) - 1
                for eqn in eqns[:-1]] + last
    
    def _gluing_equation_error(self, eqns, shapes, RHS_of_last_eqn):
        return infinity_norm(self._gluing_equation_errors(
            eqns, shapes, RHS_of_last_eqn))
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
          
