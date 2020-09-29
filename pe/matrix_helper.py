from .sage_helper import (matrix, vector, Id2, eigenvalues, get_pi,
                          complex_field, complex_I, pari, arg)

def SL2C_inverse(A):
    return A.adjugate()

def GL2C_inverse(A):
    return (1/A.det())*A.adjugate()

def normalize_vector(v):
    """Divide this non-zero vector by its L2 norm."""
    return v/v.norm()

def apply_matrix(mat, v):
    """Multiply the matrix times the vector and return the normalized result."""
    return normalize_vector(mat*v)

def vector_dist(a, b):
    """Return the L2 distance between two vectors."""
    return (a - b).norm()

def right_kernel_two_by_two(A):
    """
    For a 2x2 matrix A over an approximate field like RR or CC, find an
    element in the right kernel.
    """
    prec = A.base_ring().precision()
    epsilon = (2.0)**(-0.8*prec)
    assert A.determinant().abs() < epsilon, 'Matrix looks non-singular'
    a, b = max(A.rows(), key=lambda v: v.norm())
    v = vector([1, -a/b]) if b.abs() > a.abs() else vector([-b/a, 1])
    assert (A*v).norm() < epsilon, 'Supposed kernel vector is not in the kernel.'
    return (1/v.norm())*v

def eigenvectors(A):
    """
    Returns the eigenvectors of the matrix A that live in its field of
    definition.
    """
    eigval = eigenvalues(A)
    # For (essentially) parabolic matrices in SL(2, R), sometime the
    # eigenvalues are actually have very slight imaginary components
    # and so don't show up in eigenvalues.
    prec = A.base_ring().precision()
    epsilon = (2.0)**(-0.8*prec)
    if len(eigval) == 0:
        for e in [1, -1]:
            if (A - e*Id2).determinant().abs() < epsilon:
                eigval.append(e)
    return [right_kernel_two_by_two(A - e*Id2) for e in eigval]

def eigenvector(A):
    """
    Returns the eigenvector corresponding to the larger eigenvalue of a
    loxodromic matrix A
    """
    evalues = eigenvalues(A)
    evalue = max(evalues, key=lambda x: x.abs())
    return right_kernel_two_by_two(A - evalue*Id2)

def eigenbasis(A, B):
    """
    Given loxodromic matrices A and B, return a basis of C^2 consisting of
    one eigenvector from each.
    """
    eA = eigenvector(A)
    eB = eigenvector(B)
    return matrix([[eA[0], eB[0]], [eA[1], eB[1]]])

def conjugator_into_PSL2R(A, B):
    """
    Given loxodromic matrices A and B which lie in a common conjugate of
    PSL(2, R), return a matrix C so that C^(-1)*A*C and C^(-1)*B*C are in
    PSL(2, R) itself.
    """
    C = eigenbasis(A, B)
    AA = GL2C_inverse(C)*A*C
    BB = GL2C_inverse(C)*B*C
    a = AA[0, 1]
    b = BB[1, 0]
    if abs(a) > abs(b):
        e, f = 1, abs(a)/a
    else:
        e, f = abs(b)/b, 1

    return C * matrix(A.base_ring(), [[e, 0], [0, f]])

def fixed_point(A):
    """
    Return a complex number fixed by the linear fractional
    transformation given by a matrix A.  In the case of a parabolic,
    the fixed point will be real, unless the parabolic fixed point is
    at infinity in which case all hell breaks loose.
    """
    assert A.trace().abs() <= 2.0, 'Please make sure you have not changed the generators!'
    CC = complex_field(A.base_ring())
    x = pari('x')
    a, b, c, d = [pari(z) for z in A.list()]
    p = c*x*x + (d - a)*x - b
    if p == 0:
        return complex_I(CC)
    fp = max(p.polroots(precision=CC.precision()), key=lambda z: z.imag())
    return CC(fp)

def elliptic_rotation_angle(A):
    """Return the rotation angle of an element of PSL(2,R) at its fixed point."""
    z = fixed_point(A)
    c, d = A.list()[2:]
    derivative = 1/(c*z + d)**2
    pi = get_pi(A.base_ring())
    r = -arg(derivative)
    if r < 0:
        r = r + 2*pi
    return r/(2*pi)
