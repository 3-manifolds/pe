"""
Example:

sage -python phc_hack.py x,y "x^2 + y^2 - 1" "x-y"
[[-0.7071067811865476], [0.7071067811865476]]
"""

import sys, phc

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

def raw_solutions(variables, equations, max_err=1e-6):
    N = len(variables)//2
    ring = phc.PolyRing(variables)        
    system = phc.PHCSystem(ring, [phc.PHCPoly(ring, eqn) for eqn in equations])
    ans = []
    try:
        sols = system.solution_list()
    except phc.PHCInternalAdaException:
        return []
    for sol in sols:
        if sol.err < max_err:
            ans.append([clean_complex(z) for z in sol.point[:N]])
    return ans

if __name__ == '__main__':
    variables = sys.argv[1].split(',')
    equations = sys.argv[2:]
    print raw_solutions(variables, equations)
