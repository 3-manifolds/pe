import snappy
from snappy.snap.shapes import (pari, gen, float_to_pari, complex_to_pari,
                                pari_column_vector, prec_bits_to_dec)
from snappy.snap.shapes import (infinity_norm, pari_matrix, pari_vector_to_list,
                                enough_gluing_equations, eval_gluing_equation,
                                _within_sage, ComplexField)
if _within_sage:
    from sage.all import exp, CC

from . import Shape

def sage_complex_to_pari(z, dec_prec):
    return pari.complex( float_to_pari(z.real(), dec_prec), float_to_pari(z.imag(), dec_prec) )

def pari_set_precision(x, dec_prec):
    return pari(0) if x == 0 else pari(x).precision(dec_prec)

class GoodShapesNotFound(Exception):
    pass

class PolishedShape(Shape):
    """A refined Shape containing an arbitrarily precise solution to the
    gluing equations with a specified target value for the meridian
    holonomy.

    >>> M = snappy.Manifold('m071(0,0)')
    >>> alpha = polished_tetrahedra_shapes(M, 0, bits_prec=500)
    >>> M = snappy.Manifold('m071(7,0)')
    >>> beta = polished_tetrahedra_shapes(M, 2*CC.pi()/7, bits_prec=1000)

    """
    def __init__(self, manifold, target_holonomy, values, tolerance=1.0E-6,
                 dec_prec=None, bits_prec=200, ignore_solution_type=False):
        super(self, Shape).__init__(manifold, values, tolerance)
        if dec_prec is None:
            dec_prec = prec_bits_to_dec(bits_prec)
        else:
            bits_prec = prec_dec_to_bits(dec_prec)
        working_prec = dec_prec + 10
        target_espilon = pari_set_precision(10.0, working_prec)**-dec_prec
        det_epsilon = pari_set_precision(10.0, working_prec)**-(dec_prec//10)
        init_shapes = pari_column_vector(
            [complex_to_pari(z, working_prec) for z in self.array])
        manifold = manifold.copy()
        manifold.dehn_fill( (1, 0) ) 
        init_equations = manifold.gluing_equations('rect')
        target = pari.complex(
            pari_set_precision(target_holonomy.real(), dec_prec),
            pari_set_precision(target_holonomy.imag(), dec_prec))
        if gluing_equation_error(init_equations, init_shapes, target) > pari(0.000001):
            raise GoodShapesNotFound('Initial solution not very good')
    
        # Now begin the actual computation
        
        eqns = enough_gluing_equations(manifold)
        assert eqns[-1] == manifold.gluing_equations('rect')[-1]
    
        shapes = init_shapes 
        for i in range(100):
            errors = gluing_equation_errors(eqns, shapes, target)
            if infinity_norm(errors) < target_espilon:
                break
    
            derivative = [ [  eqn[0][i]/z  - eqn[1][i]/(1 - z)
                              for i, z in enumerate(pari_vector_to_list(shapes))] for eqn in eqns]
            derivative[-1] = [ target*x for x in derivative[-1] ]
            derivative = pari_matrix(derivative)
    
            det = derivative.matdet().abs()
            if min(det, 1/det) < det_epsilon:
                break  # Pari might crash
            gauss = derivative.matsolve(pari_column_vector(errors))
            shapes = shapes - gauss
    
        # Check to make sure things worked out ok.
        error = gluing_equation_error(init_equations, shapes, target)
        total_change = infinity_norm(init_shapes - shapes)
        if error > 1000*target_espilon or total_change > pari(0.0000001):
            raise GoodShapesNotFound('Failed to find solution')
        ans = pari_vector_to_list(shapes)
        if _within_sage:
            CC = ComplexField(bits_prec)
            self.sage = [CC(z) for z in ans]

    def gluing_equation_errors(eqns, shapes, RHS_of_last_eqn):
        last = [eval_gluing_equation(eqns[-1], shapes) - RHS_of_last_eqn]
        return [eval_gluing_equation(eqn, shapes) - 1 for eqn in eqns[:-1]] + last
    
    def gluing_equation_error(eqns, shapes, RHS_of_last_eqn):
        return infinity_norm(gluing_equation_errors(eqns, shapes, RHS_of_last_eqn))
    
    def clean_pari_complex(z, working_prec):
        epsilon = pari_set_precision(10.0, working_prec)**-(working_prec//2)
        zero = pari_set_precision(0.0, working_prec)
        r, i = z.real().abs(), z.imag().abs()
        if r < epsilon and i < epsilon:
            ans = zero
        elif r < epsilon:
            ans = pari.complex(zero, z.imag())
        elif i < epsilon:
            ans = z.real()
        else:
            ans = z
        assert (z - ans).abs() < epsilon
        return ans
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
          
