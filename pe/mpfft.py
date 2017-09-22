import numpy, math
from numpy import array, zeros, exp
from sage.all import ComplexField, pi, I

class DoubleFFT:
    """
    A bare bones, recursive FFT using ordinary double precision
    arithmetic.
    """
    def __init__(self, N):
        assert 2**int(math.log(N, 2)) == N, 'N must be a power of 2.'
        self.N = N
        # The FFT world thinks that the unit circle is oriented clockwise.
        self.roots = exp(array([-2*n*numpy.pi*1j/N for n in range(N)]))
        self.iroots = exp(array([2*n*numpy.pi*1j/N for n in range(N)]))
            
    def _fft(self, A):
        k, N = len(A), self.N
        half, stride = N // 2, N // k
        if k == 1:
            return A
        even = self._fft(A[0:k:2])
        odd = self._fft(A[1:k:2])
        result = zeros(k, dtype='complex')
        result[:k//2] = even + self.roots[0:half:stride]*odd
        result[k//2:] = even + self.roots[half:N:stride]*odd
        return result

    def _ifft(self, A):
        k, N = len(A), self.N
        half, stride = N // 2, N // k
        if k == 1:
            return A
        even = self._ifft(A[0:k:2])
        odd = self._ifft(A[1:k:2])
        result = zeros(k, dtype='complex')
        result[:k//2] = even + self.iroots[0:half:stride]*odd
        result[k//2:] = even + self.iroots[half:N:stride]*odd
        return result

    def fft(self, A):
        assert len(A) == self.N
        return self._fft(A)
    
    def ifft(self, A):
        assert len(A) == self.N
        return self._ifft(A)/self.N

class ComplexFFT:
    """
    A bare bones, recursive FFT using Sage's arbitrary precision complex
    numbers.
    """
    def __init__(self, N, precision=256):
        assert 2**int(math.log(N, 2)) == N, 'N must be a power of 2.'
        self.N = N
        self.precision = precision
        self.field = F = ComplexField(precision)
        # The FFT world thinks that the unit circle is oriented clockwise.
        self.roots = array([F((-2*k*pi*I/N).exp()) for k in range(N)],
                           dtype='O')
        self.iroots = array([F((2*k*pi*I/N).exp()) for k in range(N)],
                            dtype='O')
            
    def _fft(self, A):
        k, N = len(A), self.N
        half, stride = N // 2, N // k
        if k == 1:
            return A
        even = self._fft(A[0:k:2])
        odd = self._fft(A[1:k:2])
        result = zeros(k, dtype='O')
        result[:k//2] = even + self.roots[0:half:stride]*odd
        result[k//2:] = even + self.roots[half:N:stride]*odd
        return result

    def _ifft(self, A):
        k, N = len(A), self.N
        half, stride = N // 2, N // k
        if k == 1:
            return A
        even = self._ifft(A[0:k:2])
        odd = self._ifft(A[1:k:2])
        result = zeros(k, dtype='O')
        result[:k//2] = even + self.iroots[0:half:stride]*odd
        result[k//2:] = even + self.iroots[half:N:stride]*odd
        return result

    def fft(self, A):
        assert len(A) == self.N
        return self._fft(A)
    
    def ifft(self, A):
        assert len(A) == self.N
        return self._ifft(A)/self.N
