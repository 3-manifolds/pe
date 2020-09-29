from mpmath import mp
from sage.all import ZZ, RealField, ComplexField, block_matrix, matrix, vector

def SL2C_inverse(A):
    return A.adjugate()

def sage_matrix_to_mpmath(A):
    return mp.matrix([list(row) for row in A])

def mpmath_matrix_to_sage(A):
    entries = list(A)
    if all(isinstance(e, mp.mpf) for e in entries):
        F = RealField(mp.prec)
        entries = [F(e) for e in entries]
    else:
        F = ComplexField(mp.prec)
        entries = [F(e.real, e.imag) for e in entries]
    return matrix(F, A.rows, A.cols, entries)

def left_mult_by_adjoint(A):
    """
    Given A in GL_2, returns the 4 x 4 matrix of the left action of A*
    on M_2 with respect to the standard basis.

    >>> A = matrix(ZZ, [[1, 2], [3, 7]])
    >>> M = left_mult_by_adjoint(A); M
    [1 0 3 0]
    [0 1 0 3]
    [2 0 7 0]
    [0 2 0 7]
    >>> B = matrix(ZZ, [[-3, 4], [5, 9]])
    >>> b = vector(B.list())
    >>> (M*b).list() == (A.conjugate_transpose()*B).list()
    True
    """
    R = A.base_ring()
    B = A.conjugate_transpose()
    b1, b2, b3, b4 = B.list()
    M = matrix(R, [[b1, 0, b3, 0], [0, b1, 0, b3], [b2, 0, b4, 0], [0, b2, 0, b4]])
    return M.transpose()

def right_mult_by_inverse(A):
    """
    Given A in SL_2, returns the 4 x 4 matrix of the right action of
    A**(-1) on M_2 with respect to the standard basis.

    >>> A = matrix(ZZ, [[1, 2], [3, 7]])
    >>> M = right_mult_by_inverse(A); M
    [ 7 -3  0  0]
    [-2  1  0  0]
    [ 0  0  7 -3]
    [ 0  0 -2  1]
    >>> B = matrix(ZZ, [[-3, 4], [5, 9]])
    >>> b = vector(B.list())
    >>> (M*b).list() == (B*A.inverse()).list()
    True
    """
    R = A.base_ring()
    B = SL2C_inverse(A)
    b1, b2, b3, b4 = B.list()
    M = matrix(R, [[b1, b2, 0, 0], [b3, b4, 0, 0], [0, 0, b1, b2], [0, 0, b3, b4]])
    return M.transpose()

def nearly_diagonal(A):
    assert A.rows == A.cols == 2
    a = A[0,0]
    return mp.norm(A - mp.diag([a, a])) < 1000*mp.eps

def preserves_hermitian_form(SL2C_matrices):
    """
    >>> CC = ComplexField(100)
    >>> A = matrix(CC, [[1, 1], [1, 2]]);
    >>> B = matrix(CC, [[0, 1], [-1, 0]])
    >>> C = matrix(CC, [[CC('I'),0], [0, -CC('I')]])
    >>> ans, sig, form = preserves_hermitian_form([A, B])
    >>> ans
    True
    >>> sig
    'indefinite'
    >>> form.change_ring(ComplexField(10))
    [  0.00 -1.0*I]
    [ 1.0*I   0.00]
    >>> preserves_hermitian_form([A, B, C])
    (False, None, None)
    >>> ans, sig, form = preserves_hermitian_form([B, C])
    >>> sig
    'definite'
    """
    M = block_matrix(len(SL2C_matrices), 1,
                     [left_mult_by_adjoint(A) - right_mult_by_inverse(A) for
                      A in SL2C_matrices])

    CC = M.base_ring()
    mp.prec = CC.prec()
    RR = RealField(CC.prec())
    epsilon = RR(2)**(-int(0.8*mp.prec))
    U, S, V = mp.svd(sage_matrix_to_mpmath(M))
    S = list(mp.chop(S, epsilon))
    if mp.zero not in S:
        return False, None, None
    elif S.count(mp.zero) > 1:
        for i, A in enumerate(SL2C_matrices):
            for B in SL2C_matrices[i+1:]:
                assert (A*B - B*A).norm() < epsilon

        sig = 'indefinite' if any(abs(A.trace()) > 2 for A in SL2C_matrices) else 'both'
        return True, sig, None
    else:
        in_kernel = list(mp.chop(V.H.column(S.index(mp.zero))))
        J = mp.matrix([in_kernel[:2], in_kernel[2:]])
        iJ = mp.mpc(imag=1)*J
        J1, J2 = J + J.H, iJ + iJ.H
        J = J1 if mp.norm(J1) >= mp.norm(J2) else J2
        J = (1/mp.sqrt(abs(mp.det(J))))*J
        J = mpmath_matrix_to_sage(J)
        assert all((A.conjugate_transpose() * J * A - J).norm() < epsilon
                              for A in SL2C_matrices)
        sig = 'definite' if J.det() > 0 else 'indefinite'
        return True, sig, J


def conjugator_into_SL2R(SL2C_matrices):
    """
    Returns a matrix C in SL(2, C) so that C^-1 * M * C is
    (essentially) in SL(2, R) for all the input matrices M.
    """
    ans, sig, form = preserves_hermitian_form(SL2C_matrices)
    if ans is None:
        raise ValueError('No invariant hermitian form found')
    if sig == 'definite':
        raise ValueError('Conjugate into SU(2), not SL(2, R)')
    if sig == 'both':
        raise ValueError('This degnerate case not implemented')
    assert sig == 'indefinite'
    J = sage_matrix_to_mpmath(form)
    eigs, U = mp.eighe(J)
    C = U * mp.diag([1/mp.sqrt(abs(e)) for e in eigs])
    sq_two = mp.sqrt(2)
    sq_two_i = mp.mpc(imag=sq_two)
    S = mp.matrix([[1/sq_two_i, 1/sq_two_i], [-1/sq_two, 1/sq_two]])
    C = C * S.H
    C = (1/mp.sqrt(mp.det(C)))*C
    return mpmath_matrix_to_sage(C)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    CC = ComplexField(53)
    A = matrix(CC, [[1, 1], [1, 2]])
    B = matrix(CC, [[0, 1], [-1, 0]])
    C = matrix(CC, [[CC('I'), 1], [0, -CC('I')]])
    Bp = (A*C)*B*SL2C_inverse(A*C)
    Ap = (A*C)*A*SL2C_inverse(A*C)
