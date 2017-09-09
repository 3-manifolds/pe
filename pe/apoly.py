# -*- coding: utf-8 -*-
from __future__ import print_function
import os, tkinter
from snappy import Manifold
from .pecharvar import CircleElevation
from .shape import U1Q, PolishedShapeSet
from .input import user_input
from .plot import Plot
import numpy
from numpy import (array, matrix, dot, prod, diag, transpose, zeros, ones, eye,
                   log, exp, pi, sqrt, ceil, dtype, take, arange, sum, argmin)
from numpy.linalg import svd, norm, eig, solve, lstsq, matrix_rank

class Infinity(object):
    def __repr__(self):
        return '1/0'

try:
    from sage.all import PolynomialRing, IntegerRing, RationalField, RealField, ComplexField
    from quadFFT import QuadFFT
    ZZ = IntegerRing()
    QQ = RationalField()
    QuadC = ComplexField(114)
    QuadR = RealField(114)
    sage_poly_ring = PolynomialRing(IntegerRing(), ('M', 'L'))
    def fraction(numerator, denominator):
        if denominator:
            return QQ(numerator)/QQ(denominator)
        elif numerator:
            return Infinity()
        else:
            raise RuntimeError('0/0 is undefined.')

except ImportError:
    def no_sage(*args):
        print('Sage could not be imported')
    sage_poly_ring = no_sage
    def fraction(numerator, denominator):
        return '%d/%d'%(numerator, denominator)
    
class Apoly:
    """
    The A-polynomial of a SnapPy manifold.  

    Constructor: Apoly(mfld, order=128, gluing_form=False, denom=None, multi=False,
                       use_hints=True, verbose=True)

    <mfld>           is a manifold name recognized by SnapPy, or a Manifold instance.
    <gluing_form>    (True/False) indicates whether to find a "standard"
                     A-polynomial, or the gluing variety variant.
    <order>          must be at least twice the M-degree.  Try doubling this
                     if the coefficients seem to be wrapping.
    <denom>          Denominator for leading coefficient.  This should be
                     a string, representing a polynomial expression in H,
                     the meridian holonomy.  e.g. denom='(H*H-1)
    <multi>          If True, multiple copies of lifts are not removed, so
                     multiplicities of factors of the polynomial are computed.
    <use_hints>      Whether to check for and use hints from a hint file.
    <verbose>        Whether to print information about the computation.
    <use_quad_fft>   Whether to use 128 bit floats for the FFT.

    Methods:

    An Apoly object A is callable:  A(x,y) returns the value at (x,y).
    A.as_string(exp='^') returns a string suitable for input to a generic symbolic
                      algebra program which uses the symbol exp for exponentiation.
    A.sage()          returns a Sage polynomial with parent ring ZZ['M', 'L']
    A.show_R_longitude_evs() uses matplotlib to graph the L-projections of 
                      arcs of the elevation of the circle of radius R in the M-plane.
    A.show_T_longitude_evs() uses matplotlib to graph the L-projections
                      of components of the inverse image of the tightened
                      circle of radius T in the M-plane.
    A.show_newton(text=False) shows the newton polygon with dots.  The text
                      flag shows the coefficients.
    A.boundary_slopes() returns the boundary slopes detected by the character
                      variety.
    A.save(basename=None, dir='polys', with_hint=True, twist=0)
                      Saves the polynomial in a .apoly or .gpoly text file for
                      input to a symbolic computation program.  The directory
                      can be overridden by specifying dir. Saves the parameters
                      in a .hint file unless with_hint==False.  Assumes that the
                      preferred longitude is LM^twist, where L,M are the SnapPea
                      meridian and longitued
    A.verify() runs various consistency checks on the polynomial.

    An Apoly object prints itself as a matrix of coefficients.
  """
    def __init__(self, mfld, order=128, gluing_form=False,
                 radius=1.02, denom=None, multi=False, use_hints=True, verbose=True,
                 apoly_dir='apolys', gpoly_dir='gpolys', base_dir='PE_base_fibers',
                 hint_dir='hints', use_quad_fft=False):
        if isinstance(mfld, Manifold):
            self.manifold = mfld
            self.mfld_name = mfld.name()
        else:
            self.mfld_name = mfld
            self.manifold = Manifold(mfld)
        self.gluing_form = gluing_form
        self.verbose = verbose
        self.apoly_dir = apoly_dir
        self.gpoly_dir = gpoly_dir
        self.base_dir = base_dir
        self.hint_dir = hint_dir
        self.use_quad_fft = use_quad_fft
        options = {'order'        : order,
                   'denom'        : denom,
                   'multi'        : multi,
                   'radius'       : radius,
                   'use_quad_fft' : use_quad_fft
                   }
                   # 'apoly_dir'   : apoly_dir,
                   # 'gpoly_dir'   : gpoly_dir,
                   # 'base_dir'    : base_dir,
                   # 'hint_dir'    : hint_dir}
        if use_hints:
            self._print("Checking for hints ... ", end='')
            hintfile = os.path.join(self.hint_dir, self.mfld_name+'.hint')
            if os.path.exists(hintfile):
                self._print("yes!")
                exec(open(hintfile).read())
                options.update(hint)
                self._print('Using: radius=%f; order=%d; denom=%s%s.'%(
                    hint['radius'], hint['order'], hint['denom'],
                    ' with quad precision' if hint['use_quad_fft'] else ''))
            else:
                print("nope.")
        self.order = N = options['order']
        self.denom = options['denom']
        if self.denom:
            exec('f = lambda H : %s'%self.denom)
            self.denom_function = f
        self.multi = options['multi']
        self.radius = options['radius']
        if use_quad_fft:
            self.quad_fft = QuadFFT(self.order)
        filename = self.manifold.name()+'.base'
        saved_base_fiber = os.path.join(self.base_dir, filename)
        self.elevation = CircleElevation(
            self.manifold,
            order=self.order,
            radius=self.radius,
            base_dir=self.base_dir,
            verbose=self.verbose)
        if self.elevation.failed:
            print("Warning: Failed to elevate the R-circle.  This Apoly is incomplete.")
            return
        if self.gluing_form:
            vals = array([track for track in self.elevation.R_longitude_holos])
        else:
            if self.use_quad_fft:
                self.elevation.polish_R_longitude_vals()
                vals = array(self.elevation.polished_R_longitude_evs)
            else:
                vals = array(self.elevation.R_longitude_evs)
        self.degree = len(vals)
        if multi == False:
            self.multiplicities, vals = self.demultiply(vals)
        self.reduced_degree = len(vals)
        self._compute_all(vals)
        if self.height > float(2**52):
            print("Coefficients overflowed.")
        self._print("Noise levels: ")
        for level in self.max_noise:
            self._print(level)
        if max(self.max_noise) > 0.1:
            self._print('Failed to find integer coefficients with tolerance 0.1')
            return
        self._print('Computing the Newton polygon ... ', end='')
        self.compute_newton_polygon()
        self._print('done.')
        
    def __call__(self, M, L):
        result = 0
        rows, cols = self.coefficients.shape
        for i in range(rows):
            Lresult = 0
            for j in range(cols):
                Lresult = Lresult*L + self.coefficients[-1-i][-1-j]
            result = result*M + Lresult
        return result
    
    def __repr__(self):
        return 'A-polynomial of %s'%self.manifold

    def __str__(self):
        digits = 2 + int(ceil(log(self.height)/log(10)))
        width = len(self.coefficients[0])
        format = '[' + ('%' + str(digits) + '.0f')*width + ']\n'
        result = ''
        for row in self.coefficients:
            result += format%tuple(row + 0.)
        return result

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _compute_all(self, vals):
        """
        Use a discrete Fourier transform to compute the A-polynomial from the
        longitude eigenvalues (or holonomies).  We are viewing the A-polynomial
        as a polynomial in L with coefficients in QQ[M].  For a number z on the
        R-circle, we find a monic polynomial in L vanishing on the values taken
        by L at those points of the A-curve where M takes the value z.  Thus we
        get a polynomial in L whose coefficients are polynomials in M vanishing
        on the A-curve.  To compute the integer coefficients of these
        M-polynomials we renormalize to the unit circle and use the inverse FFT.
        (Note: this produces a monic polynomial in L, but the A-polynomial is
        not monic when there is an ideal point of the character variety which
        is associated with a meridian boundary slope.  The denom parameter is
        used to resolve this issue.)
        """
        self.sampled_roots = vals
        self.sampled_coeffs = self.symmetric_funcs(vals)
        if self.use_quad_fft:
            radius = QuadR(self.radius)
            ifft = self.quad_fft.ifft
            def real(A):
                return array([z.real() for z in A])
        else:
            radius = self.radius
            ifft = numpy.fft.ifft
            def real(A):
                return array([z.real for z in A])
        if self.denom:
            if self.use_quad_fft:
                circle = [radius*U1Q(-n, self.order, precision=114) for n in range(self.order)]
                self.D = D = array([self.denom_function(z) for z in circle])
            else:
                self.D = D = array([self.denom_function(z) for z in self.elevation.R_circle])
            self.raw_coeffs = array([ifft(x*D) for x in self.sampled_coeffs])
        else:
            self.raw_coeffs = array([ifft(x) for x in self.sampled_coeffs])
        # Renormalize the coefficients, to adjust for the circle radius
        N = self.order
        if N%2 == 0:
            powers = -array(list(range(1+N//2))+list(range(1-N//2, 0)))
        else:
            powers = -array(list(range(1+N//2))+list(range(-(N//2), 0)))
        renorm = array([radius**n for n in powers])
        self.normalized_coeffs = self.raw_coeffs*renorm
        self.int_coeffs = array([map(round, real(x)) for x in self.normalized_coeffs])
        self.height = max([max(abs(x)) for x in self.int_coeffs])
        self.noise = self.normalized_coeffs.real - self.int_coeffs
        self.max_noise = [max(abs(x)) for x in self.noise]
        self.shift = self.find_shift()
        self._print('Shift is %s'%self.shift)
        if self.shift is None:
            raise ValueError('Could not compute the shift. '
                             'Coefficients may be wrapping.  '
                             'If so, a larger order might help.')
        C = self.int_coeffs.transpose()
        coefficient_array =  take(C, arange(len(C))-self.shift, axis=0)
        rows, cols = coefficient_array.shape
        while rows > 0:
            if max(abs(coefficient_array[rows-1])) > 0:
                break
            rows -= 1
        self.coefficients = coefficient_array[:rows]
        
    def compute_newton_polygon(self):
        power_scale = (1,1) if self.gluing_form else (1,2) 
        self.newton_polygon = NewtonPolygon(self.as_dict(), power_scale)

    def recompute(self):
        """
        Recompute A after changing attributes.
        """
        self._compute_all(array(self.elevation.R_longitude_evs))    
        self.compute_newton_polygon()

    def help(self):
        print(self.__doc__)

    def symmetric_funcs(self, evs):
        """
        Given a numpy 2D array whose rows are L-eigenvalues sampled on a circle,
        return a 2D array whose rows are the elementary symmetric functions of
        the roots.

        Before computing the elementary symmetric functions, each column is
        sorted in descending size, to avoid 'catastrophic cancellation'. This
        means, for example (using decimal floating point with 3 digits) that
        we want to compute
           .101E3 - .100E3 - .999E0 = .1E-2 == .001
        rather than
           .101E3 - .999E0 - .100E3 "=" .1E1 == 1.0
        """
        for n in range(evs.shape[1]):
            evs[:,n] = sorted(evs[:,n], key=lambda x: -abs(x))
        coeffs = [0, ones(evs[0].shape, evs.dtype)]
        for root in evs:
            for i in range(1, len(coeffs)):
                coeffs[-i] = -root*coeffs[-i] + coeffs[-1-i]
            coeffs.append(ones(evs[0].shape, evs.dtype))
        return coeffs[1:]

    def demultiply(self, eigenvalues):
            multiplicities = []
            sdr = [] #system of distinct representatives
            multis = [1]*len(eigenvalues)
            for i in range(len(eigenvalues)):
                unique = True
                for j in range(i+1,len(eigenvalues)):
                    # If this row is the same as a lower row, do not
                    # put it in the sdr.  Just increment the multiplicity
                    # of the lower row.
                    if max(abs(eigenvalues[i] - eigenvalues[j])) < 1.0E-6:
                        unique = False
                        multis[j] += multis[i]
                        break
                if unique:
                    sdr.append(i)
                    multiplicities.append((i, multis[i]))
            return multiplicities, take(eigenvalues, sdr, 0)

    def find_shift(self):
       rows, cols = self.normalized_coeffs.shape
       shifts = [0]
       #start from the top and search for the last row above the middle
       #whose left-most non-zero entry is +-1.
       for i in range(rows):
          for j in range(1, 1 + cols//2):
             if abs(abs(self.normalized_coeffs[i][-j]) - 1.) < .01:
                 shifts.append(j)
       return max(shifts)

# Should have a monomial class, and generate a list of monomials here not a string
    def monomials(self):
        rows, cols = self.coefficients.shape
        monomials = []
        for j in range(cols):
            for i in range(rows):
                if self.gluing_form:
                    m,n = 2*i, 2*j
                else:
                    m,n = 2*i, j
                a = int(self.coefficients[i][j])
                if a != 0:
                    if i > 0:
                        if j > 0:
                            monomial = '%d*(M^%d)*(L^%d)'%(a,m,n)
                        else:
                            monomial = '%d*(M^%d)'%(a,m)
                    else:
                        if j > 0:
                            monomial = '%d*(L^%d)'%(a,n)
                        else:
                            monomial = '%d'%a
                    monomials.append(monomial)
        return monomials

# Should use the list of monomials to generate the dict
    def as_dict(self):
        rows, cols = self.coefficients.shape
        result = {}
        for j in range(cols):
            for i in range(rows):
                if self.gluing_form:
                    m,n = 2*i, 2*j
                else:
                    m,n = 2*i, j
                coeff = int(self.coefficients[i][j])
                if coeff:
                    result[(m,n)] = coeff
        return result

    def break_line(self, line):
        marks = [0]
        start = 60
        while True:
            mark = line.find('+', start)
            if mark == -1:
                break
            marks.append(mark)
            start = mark+60
        lines = []
        for i in range(len(marks) - 1):
            lines.append(line[marks[i]:marks[i+1]])
        lines.append(line[marks[-1]:])
        return '\n    '.join(lines)
    
    def as_string(self, exp='^'):
        polynomial_string = ('+'.join(self.monomials())).replace('+-','-')
        return polynomial_string.replace('^', exp)

    def sage(self):
        return sage_poly_ring(self.as_string())
    
    # could do this by sorting the monomials
    def as_Lpolynomial(self, name='A', twist=0):
        terms = []
        rows, cols = self.coefficients.shape
        #We are taking the true longitude to be L*M^twist.
        #So we change variables by L -> M^(-twist)*L.
        #Then renormalize so the minimal power of M is 0.
        minexp = 2*rows
        for j in range(cols):
            for i in range(rows):
                if self.coefficients[i][j]:
                    break
            minexp = min(2*i - j*twist, minexp)
        for j in range(cols):
            if self.gluing_form:
                n = 2*j
            else:
                n = j
            monomials = []
            for i in range(rows):
                m = 2*i
                a = int(self.coefficients[i][j])
                if a != 0:
                    if i > 0:
                        monomial = '%d*M^%d'%(a,m)
                    else:
                        monomial = '%d'%a
                    monomials.append(monomial.replace('^1 ',' '))
            if monomials:
                p = - n*twist - minexp
                if p:
                    P = '%d'%p
                    if p < 0:
                        P = '('+P+')'
                    if n > 0:
                        term = '+ (L^%d*M^%s)*('%(n,P) + ' + '.join(monomials) + ')'
                    else:
                        term = '(M^%s)*('%P + ' + '.join(monomials) + ')'
                else:
                    if n > 0:
                        term = '+ (L^%d)*('%n + ' + '.join(monomials) + ')'
                    else:
                        term = '(' + ' + '.join(monomials) + ')'
                term = self.break_line(term)
                terms.append(term.replace('+ -','- '))
        return name + ' :=\n' + '\n'.join(terms)

    def save(self, basename=None, dir=None, with_hint=True, twist=0):
        if dir == None:
            if self.gluing_form:
                poly_dir = self.gpoly_dir
                hint_dir = self.hint_dir
                ext = '.gpoly'
            else:
                poly_dir = self.apoly_dir
                hint_dir = self.hint_dir
                ext = '.apoly'
        for dir in (poly_dir, hint_dir):
            if not os.path.exists(dir):
                cwd = os.path.abspath(os.path.curdir)
                newdir = os.path.join(cwd,dir)
                response = user_input("May I create a directory %s?(y/n)"%newdir)
                if response.lower()[0] != 'y':
                    sys.exit(0)
                os.mkdir(newdir)
        if basename == None:
            basename = self.mfld_name
        polyfile_name = os.path.join(poly_dir, basename + ext)
        hintfile_name = os.path.join(hint_dir, basename + '.hint')
        polyfile = open(polyfile_name,'w')
        if self.gluing_form:
            lhs = 'G_%s'%basename
        else:
            lhs = 'A_%s'%basename
        polyfile.write(self.as_Lpolynomial(name=lhs, twist=twist))
        polyfile.write(';\n')
        polyfile.close()
        if with_hint:
            self.elevation.save_hint(
                directory=self.hint_dir,
                extra_options={
                    'denom': self.denom,
                    'multi': self.multi,
                    'use_quad_fft': self.use_quad_fft
                })
            
    def boundary_slopes(self):
        return [s.sage() for s in self.newton_polygon.lower_slopes]
        
    def show_R_longitude_evs(self):
        self.elevation.show_R_longitude_evs()

    def show_T_longitude_evs(self):
        if not self.elevation.T_circle:
            self.tighten()
        self.elevation.show_T_longitude_evs()

    def show_coefficients(self):
        plot = Plot(self.normalized_coeffs.real)

    def show_noise(self):
        plot = Plot(self.noise)

    def show_imag_noise(self):
        plot = Plot(self.normalized_coeffs.imag)

    def show_newton(self, text=False):
        V = PolyViewer(self.newton_polygon, title=self.mfld_name)
        V.show_sides()
        if text:
            V.show_text()
        else:
            V.show_dots()

    def show_R_volumes(self):
        H = self.elevation
        Plot(H.compute_volumes(H.R_fibers))

    def show_T_volumes(self):
        H = self.elevation
        Plot(H.compute_volumes(H.T_fibers))

    def tighten(self, T=1.0):
        self.elevation.tighten(T)

    def verify(self):
        noise_ok = True
        symmetry = True
        sign = None
        self._print('Checking max noise level: ', end=' ')
        self._print(max(self.max_noise))
        if max(self.max_noise) > .3:
            noise_ok = False
            self._print('Failed')
        self._print('Checking for reciprocal symmetry ... ', end=' ')
        if max(abs(self.coefficients[0] - self.coefficients[-1][-1::-1]))==0:
            sign = -1.0
        elif max(abs(self.coefficients[0] + self.coefficients[-1][-1::-1]))==0:
            sign = 1.0
        else:
            self._print('Failed!')
            symmetry = False
        if sign:
            for i in range(len(self.coefficients)):
                maxgap = max(abs(self.coefficients[i] +
                              sign*self.coefficients[-i-1][-1::-1]))
                if maxgap > 0:
                    self._print('Failed! gap = %d'%maxgap)
                    symmetry = False
        if symmetry:
            self._print('OK.')
        result = noise_ok and symmetry 
        if result:
            self._print('Passed!')
        return result

# This won't work with bad fibers.    
#    def tighten(self, T=1.0):
#      self.elevation.tighten(T)
#      roots = [array(x) for x in self.elevation.T_longitude_evs]
#      self.T_sampled_coeffs = self.symmetric_funcs(roots)
#      self.T_raw_coeffs = array([ifft(x) for x in self.T_sampled_coeffs])

class Slope:
    def __init__(self, xy, power_scale=(1,1)):
        x, y = xy
        x, y = x*power_scale[0], y*power_scale[1]
        if x == 0:
            if y == 0:
                raise ValueError('gcd(0,0) is undefined.')
            else:
                gcd = abs(y)
        else:
            x0 = abs(x)
            y0 = abs(y)
            while y0 != 0:
                r = x0%y0
                x0 = y0
                y0 = r
            gcd = x0
        if x < 0:
            x, y = -x, -y
        self.x = x//gcd
        self.y = y//gcd
        
    def __eq__(self, other):
        return self.y*other.x == other.y*self.x

    def __lt__(self, other):
        return self.y*other.x < other.y*self.x

    def __repr__(self):
        return '%d/%d'%(self.y, self.x)

    def sage(self):
        return fraction(self.y, self.x)
    
class NewtonPolygon:
      def __init__(self, coeff_dict, power_scale=(1,1)):
          # Clean up weird sage tuples
          self.coeff_dict = {}
          for (degree, coefficient) in list(coeff_dict.items()):
              self.coeff_dict[tuple(degree)] = coefficient
          # The X-power is the y-coordinate!
          self.support = [(x[1], x[0]) for x in list(coeff_dict.keys())]
          self.support.sort()
          self.lower_slopes = []
          self.upper_slopes = []
          self.lower_vertices = []
          self.upper_vertices = []
          self.newton_sides = {}
          self.find_vertices()

      def slope(self, v, w):
          return Slope((w[0]-v[0], w[1]-v[1]))

      def find_vertices(self):
          last = self.support[0]
          T = []
          B = [last]
          for e in self.support[1:]:
              if e[0] != last[0]:
                  T.append(last)
                  B.append(e)
              last = e
          T.append(last)
          if T[0] != B[0]:
              self.lower_slopes.append(Slope((0,1)))
              self.upper_vertices.append(T[0])
          n = 0
          while n < len(B) - 1:
              self.lower_vertices.append(B[n])
              slopes = [(self.slope(B[n], B[k]), -k) for k in range(n+1,len(B))]
              slope, m = min(slopes)
              self.lower_slopes.append(slope)
              if slope.y < 0:
                  newton_side = [B[n]]
                  for s, j in slopes:
                      if s == slope and -j <= -m:
                          newton_side.append(B[-j])
                  self.newton_sides[(slope.x,slope.y)] = newton_side
              n = -m
          n = 0
          while n < len(T) - 1:
              slope, m = max([(self.slope(T[n], T[k]), k) for k in range(n+1,len(T))])
              self.upper_slopes.append(slope)
              self.upper_vertices.append(T[m])
              n = m
          if T[-1] != B[-1]:
              self.upper_slopes.append(Slope((0,1)))
              self.lower_vertices.append(B[-1])

      def side_dicts(self):
          result = {}
          for slope, side in list(self.newton_sides.items()):
              side_dict = {}
              for i, j in side:
                  side_dict[(j,i)] = self.coeff_dict[(j,i)]
              result[slope] = side_dict
          return result
    
      def puiseux_expansion(self):
          result = []
          for slope, side_dict in list(self.side_dicts().items()):
              P = sage_2poly(dict)
              m, n = slope
              result.append(P(t**n,t**m))

class PolyViewer:
      def __init__(self, newton_poly, title=None, scale=None, margin=50):
          self.NP = newton_poly
          self.columns = 1 + self.NP.support[-1][0]
          self.rows = 1 + max([d[1] for d in self.NP.support])
          if scale == None:
                scale = 600/max(self.rows, self.columns)
          self.scale = scale
          self.margin = margin
          self.width = (self.columns - 1)*self.scale + 2*self.margin
          self.height = (self.rows - 1)*self.scale + 2*self.margin
          self.window = tkinter.Tk()
          if title:
              self.window.title(title)
          self.window.wm_geometry('+400+20')
          self.canvas = tkinter.Canvas(self.window,
                                  bg='white',
                                  height=self.height,
                                  width=self.width)
          self.canvas.pack(expand=True, fill=tkinter.BOTH)
          self.font = ('Helvetica','18','bold')
          self.dots=[]
          self.text=[]
          self.sides=[]

          self.grid = (
              [ self.canvas.create_line(
              0, self.height - self.margin - i*scale,
              self.width, self.height - self.margin - i*scale,
              fill=self.gridfill(i))
                        for i in range(self.rows)] +
              [ self.canvas.create_line(
              self.margin + i*scale, 0,
              self.margin + i*scale, self.height,
              fill=self.gridfill(i))
                        for i in range(self.columns)])
#          self.window.mainloop()

      def write_psfile(self, filename):
            self.canvas.postscript(file=filename)
            
      def gridfill(self, i):
          if i:
              return '#f0f0f0'
          else:
              return '#d0d0d0'
          
      def point(self, pair):
          i,j = pair
          return (self.margin+i*self.scale,
                  self.height - self.margin - j*self.scale)
      
      def show_dots(self):
          r = 1 + self.scale/20
          for i, j in self.NP.support:
              x,y = self.point((i,j))
              self.dots.append(self.canvas.create_oval(
                  x-r, y-r, x+r, y+r, fill='black'))

      def erase_dots(self):
          for dot in self.dots:
              self.canvas.delete(dot)
          self.dots = []

      def show_text(self):
          r = 2 + self.scale/20
          for i, j in self.NP.support:
              x,y = self.point((i,j))
              self.sides.append(self.canvas.create_oval(
                  x-r, y-r, x+r, y+r, fill='black'))
              self.text.append(self.canvas.create_text(
                  2*r+x,-2*r+y,
                  text=str(self.NP.coeff_dict[(j,i)]),
                  font=self.font,
                  anchor='c'))
              
      def erase_text(self):
            for coeff in self.text:
                  self.canvas.delete(coeff)
            self.text=[]

      def show_sides(self):
            r = 2 + self.scale/20
            first = self.NP.lower_vertices[0]
            x1, y1 = self.point(first)
            upper = list(self.NP.upper_vertices)
            upper.reverse()
            vertices = self.NP.lower_vertices + upper + [first]
            for vertex in vertices:
                  self.sides.append(self.canvas.create_oval(
                        x1-r, y1-r, x1+r, y1+r, fill='red'))
                  x2, y2 = self.point(vertex)
                  self.sides.append(self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill='red'))
                  x1, y1 = x2, y2

      def erase_sides(self):
            for object in self.sides:
                  self.canvas.delete(object)
            self.sides=[]

class Permutation(dict):
    def orbits(self):
        points = set(self.keys())
        orbits = []
        while points:
            first = n = points.pop()
            orbit = [first]
            while True:
                n = self[n]
                if n == first:
                    orbits.append(orbit)
                    break
                else:
                    points.remove(n)
                    orbit.append(n)
        return orbits

winding = lambda x : (sum(log(x[1:]/x[:-1]).imag) + log(x[0]/x[-1]).imag)/(-2*pi)
Apoly.sage_poly = lambda self : sage_2poly(self.as_dict(), ring=QQ['M','L'])

#    PolyRelation.sage_poly = lambda self : sage_poly(self.as_dict())

#M = Manifold('4_1')
#F = Fiber((-0.991020658402+0.133708842719j),
#          [
#           ShapeVector(array([
#            6.18394729421744E-01+5.14863122901458E-02j,
#            6.18394729421744E-01-5.14863122901458E-02j], dtype=DTYPE)),
#           ShapeVector(array([
#            -1.57365927858202E+00+3.47238981119960E-01j,
#            -1.57365927858202E+00-3.47238981119960E-01j], dtype=DTYPE))
#          ])
#begin = time.time()
#B = Elevation(M, F)
#print time.time()-begin
#Z = array(M.tetrahedra_shapes('rect'))
#print B.run_newton(Z, 1j)
