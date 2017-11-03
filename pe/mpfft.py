import numpy, math
from numpy import array, zeros, exp, take
from sage.all import ComplexField, pi, I, factor

class BaseFFT:
    """
    Base class for a bare bones, recursive FFT.  Accepts array
    lengths of the form 2^m*3^n.
    """
    def __init__(self, N):
        factors = set(int(f[0]) for f in factor(N))
        assert factors.issubset(set((2,3))), 'N must be 2^m*3^n.'
        self.N = N
        self._initialize()
        
    def _initialize(self):
        # Subclasses override this method to compute roots of unity.
        # Remember that the FFT world orients the unit circle clockwise.
        pass
    
    def _zeros(self, k):
        # Subclasses override this method to return an array of zeros. 
        pass
        
    def _fft(self, A, invert):
        # The only difference between the fft and the inverse fft
        # is which generator is chosen for the group of roots of 1.
        # This choice is determined by the boolean 'invert'.
        k, N = len(A), self.N
        if k == 1:
            return A
        stride = N // k
        if invert:
            zhat, z2hat = self.izhat, self.iz2hat
        else:
            zhat, z2hat = self.zhat, self.z2hat
        result = self._zeros(k)
        if k%2 == 0:
            even = self._fft(A[0:k:2], invert)
            odd = self._fft(A[1:k:2], invert)
            result[:k//2] = even + zhat[0:N//2:stride]*odd
            result[k//2:] = even + zhat[N//2:N:stride]*odd
            return result
        elif k%3 == 0:
            zeroth = self._fft(A[0:k:3], invert)
            oneth = self._fft(A[1:k:3], invert)
            twoth = self._fft(A[2:k:3], invert)
            result[:k//3] = (zeroth +
                zhat[0:N//3:stride]*oneth +
                z2hat[0:N//3:stride]*twoth)
            result[k//3:2*k//3:] = (zeroth +
                zhat[N//3:2*N//3:stride]*oneth +
                z2hat[N//3:2*N//3:stride]*twoth)
            result[2*k//3:] = (zeroth +
                zhat[2*N//3::stride]*oneth +
                z2hat[2*N//3::stride]*twoth)
            return result

    def fft(self, A):
        assert len(A) == self.N
        return self._fft(A, False)
    
    def ifft(self, A):
        assert len(A) == self.N
        return self._fft(A, True)/self.N

class DoubleFFT(BaseFFT):
    """
    A bare bones, recursive FFT using ordinary double precision
    arithmetic.  Accepts array lengths of the form 2^m*3^n.
    """
        
    def _initialize(self):
        N = self.N
        self.zhat = exp(array([-2*n*numpy.pi*1j/N for n in range(N)]))
        self.izhat = exp(array([2*n*numpy.pi*1j/N for n in range(N)]))
        self.z2hat = self.zhat.take(xrange(0, 2*N, 2), mode='wrap')
        self.iz2hat = self.izhat.take(xrange(0, 2*N, 2), mode='wrap')
        
    def _zeros(self, k):
        return zeros(k, dtype='complex')

class ComplexFFT(BaseFFT):
    """
    A bare bones, recursive FFT using Sage's arbitrary precision complex
    numbers.   Accepts array lengths of the form 2^m*3^n.
    """
    def __init__(self, N, precision=256):
        self.precision = precision
        BaseFFT.__init__(self, N)
        
    def _initialize(self):
        N, F = self.N, ComplexField(self.precision) 
        self.zhat = array([F((-2*k*pi*I/N).exp()) for k in range(N)],
                           dtype='O')
        self.izhat = array([F((2*k*pi*I/N).exp()) for k in range(N)],
                            dtype='O')
        self.z2hat = self.zhat.take(xrange(0, 2*N, 2), mode='wrap')
        self.iz2hat = self.izhat.take(xrange(0, 2*N, 2), mode='wrap')

    def _zeros(self, k):
        return zeros(k, dtype='O')
