# -*- coding: utf-8 -*-
"""
Defines the main class PECharVariety.

A PECharVariety object represents a Peripherally Elliptic Character
Variety.  Each PECHarVariety oject manages a CircleElevation
object, which represent a family of fibers for the meridian holonomy
map lying above a circle in the compex plane.
"""
from __future__ import print_function
from collections import OrderedDict
from numpy import arange, array, float64, log, pi, angle
from snappy import Manifold, ManifoldHP
from spherogram.graphs import Graph
from .elevation import CircleElevation
from .fiber import Fiber
from .fibrator import Fibrator
from .point import PEPoint
from .shape import PolishedShapeSet, U1Q
from .plot import Plot
from .complex_reps import PSL2CRepOf3ManifoldGroup
from .real_reps import PSL2RRepOf3ManifoldGroup
import sys, os


class PEArc(list):
    """
    A list of pillowcase points lying on an arc of the PECharVariety,
    subclassed to allow for additional attributes and utility methods.
    """
    def __init__(self, *args, **kwargs):
        self.arctype = kwargs.pop('arctype', 'arc')
        super(PEArc, self).__init__(*args)

    def add_info(self, elevation, arctype='arc'):
        n, m = self.first_index = self[0].index
        self.first_shape = elevation.T_fibers[n].shapes[m]
        n, m = self.last_index = self[-1].index
        self.last_shape = elevation.T_fibers[n].shapes[m]

    def add_gap(self, L, M_args, n):
        """
        Add gaps where the arc wraps around the vertical cut edge of the
        pillowcase (L=0 or L=1).  Wrapping is deemed to have occurred
        if the next dx is more than 0.5 and has opposite sign to the
        previous dx.
        """
        assert len(self) > 1
        last_L = self[-1].real
        last_dx = self[-1].real - self[-2].real
        dx = L - last_L
        if abs(dx) > 0.5:
            if last_dx > 0 and dx < 0:   # wrapping past L = 1
                length = 1.0 + dx
                interp = ((1.0 - last_L)*M_args[n] + L*M_args[n-1])/length
                self.append(PEPoint(1.0, interp, leave_gap=True))
                self.append(PEPoint(0.0, interp))
            elif last_dx < 0 and dx > 0: # wrapping past L = 0
                length = 1.0 - dx
                interp = (last_L*M_args[n] + (1.0 - L)*M_args[n-1])/length
                self.append(PEPoint(0.0, interp, leave_gap=True))
                self.append(PEPoint(1.0, interp))

class PECharVariety(object):
    """
    An object representating the PE Character Variety of a 3-manifold.
    """
    def __init__(self, manifold, order=169, radius=1.02,
                 elevation=None, base_dir='PE_base_fibers', hint_dir='hints',
                 ignore_saved=False):
        # The default order is 13^2 because 13th roots of 1 are rarely singularities.
        self.base_dir = base_dir
        if not isinstance(manifold, (Manifold, ManifoldHP)):
            manifold = Manifold(manifold)
        self.radius = radius
        self.order = order
        if elevation is None:
            self.elevation = CircleElevation(
                manifold,
                order=order,
                radius=radius,
                base_dir=base_dir,
                hint_dir=hint_dir,
                ignore_saved=ignore_saved)
        else:
            self.elevation = elevation
        # Save the manifold here, because it may have been replaced by a saved manifold.
        self.manifold = self.elevation.manifold
        self.elevation.tighten()

    def __getitem__(self, index):
        """
        Return the indexed fiber or shape from this PECharVariety's elevation.
        """
        return self.elevation[index]

    def build_arcs(self, show_group=False):
        """
        Find the arcs in the pillowcase projection of this PE Character Variety.
        Iterates through the arcs in T_longitude_evs, extracting the subarcs
        where the longitude eigenvalue lies on the unit circle.  These arcs become
        vertices of a "curve graph".  The arcs are parametrized by decreasing values
        of the meridian eigenvalue, so they run from a local maximum to a local
        minimum of the meridian eigenvalue.  These extrema become the edges of the
        curve graph.  So the components of the curve graph correspond to closed curves
        or arcs joining corners in the pillowcase picture.  (Note that these arcs
        and curves have multiplicities!)
        """
        self.arcs = []
        elevation = self.elevation
        delta_M = -0.5/self.order
        M_args = arange(0.5, 0.0, delta_M, dtype=float64)
        for m, track in enumerate(elevation.T_longitude_evs):
            arc = PEArc()
            marker = ''
            failed = elevation.tighten_failures.get(m, set())
            for n, ev in enumerate(track):
                if n in failed:
                    # Skip over this point since we weren't able to tighten it.
                    continue
                # Is the longitude eigenvalue on the unit circle?
                if ev and .99999 < abs(ev) < 1.00001:
                    if show_group:
                        shape = elevation.T_fibers[n].shapes[m]
                        if shape.in_SU2():
                            marker = '.'
                        elif shape.has_real_traces():
                            marker = 'D'
                        else:
                            marker = 'x'
                    L = (log(ev).imag/(2*pi))%1.0
                    if len(arc) > 1:
                        arc.add_gap(L, M_args, n)
                    arc.append(PEPoint(L, M_args[n], marker=marker, index=(n, m)))
                else:
                    if len(arc) > 1:
                        arc.add_info(elevation)
                        self.arcs.append(arc)
                    # start a new arc
                    arc = PEArc()
            # If the entire track consists of real reps, we end up here with
            # a non-empty arc.
            if arc:
                arc.add_info(elevation)
                self.arcs.append(arc)
        self.curve_graph = curve_graph = Graph([], list(range(len(self.arcs))))
        self.add_extrema()
        # build the color dict
        self.colors = OrderedDict()
        for n, component in enumerate(curve_graph.components()):
            for m in component:
                self.colors[m] = n
        # Clean up endpoints at the corners of the pillowcase.
        for arc in self.arcs:
            try:
                if abs(arc[1] - arc[0]) > 0.25:
                    if arc[0].imag > 0.45 and arc[1].imag < 0.05:
                        arc[0] = arc[0] - 0.5j
                    elif arc[0].imag < 0.05 and arc[1].imag > 0.45:
                        arc[0] = arc[0]+ 0.5j
                    if arc[0].real > 0.9 and arc[1].real < 0.1:
                        arc[0] = arc[0] - 1.0
                    elif arc[0].real < 0.1 and arc[1].real > 0.9:
                        arc[0] = arc[0] + 1.0
                if abs(arc[-1] - arc[-2]) > 0.25:
                    if abs(arc[-2].imag) < 0.05 and arc[-1].imag > 0.45:
                        arc[-1] = arc[-1] - 0.5j
                    elif arc[-2].imag > 0.45 and arc[-1].imag < 0.05:
                        arc[-1] = arc[-1] + 0.5j
                    if arc[-2].real < 0.1 and arc[-1].real > 0.9:
                        arc[-1] = arc[-1] - 1.0
                    elif arc[-2].real > 0.9 and arc[-1].real < 0.1:
                        arc[-1] = arc[-1] + 1.0
            except TypeError:
                pass

    def add_extrema(self):
        """
        The pillowcase arcs, as computed, are monotone decreasing in the
        argument of the meridian.  The actual components of the
        pillowcase picture are unions of these monotone arcs joined at
        critical points of the argument of the meridian.  This method
        attempts to compute which arcs should be joined at maxima or
        minima, and adds the "cups" and "caps" to the pillowcase picture.
        """
        # Caps
        arcs = [a for a in self.arcs if a.arctype == 'arc']
        arcs.sort(key=lambda x: x[0].imag, reverse=True)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[0].imag == arcs[0][0].imag:
                level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].first_shape.dist(a.first_shape)
                                   for a in level[1:]])
                cap_pair = [level.pop(0), level.pop(distances.argmin())]
                cap_pair.sort(key=lambda a: a[0].real)
                left, right = cap_pair
                if .01 < right[0].imag < .49:
                    cap_arc = PEArc(arctype='cap')
                    cap_arc.append(left[0])
                    if left[1].real > left[0].real and right[1].real < right[0].real:
                        # \ / -- the cap wraps
                        cap_arc.append(PEPoint(0.0, left[0].imag, leave_gap=True))
                        cap_arc.append(PEPoint(1.0, left[0].imag))
                        cap_arc.append(right[0])
                    else:
                        cap_arc.append(right[0])
                    n = len(self.arcs)
                    self.arcs.append(cap_arc)
                    self.curve_graph.add_edge(self.arcs.index(left), n)
                    self.curve_graph.add_edge(n, self.arcs.index(right))
        # Cups
        arcs = [a for a in self.arcs if a.arctype == 'arc']
        arcs.sort(key=lambda x: x[-1].imag)
        while arcs:
            arc = arcs.pop(0)
            level = [arc]
            while arcs and arc[-1].imag == arcs[0][-1].imag:
                level.append(arcs.pop(0))
            while len(level) > 1:
                distances = array([level[0].last_shape.dist(a.last_shape)
                                   for a in level[1:]])
                cup_pair = [level.pop(0), level.pop(distances.argmin())]
                cup_pair.sort(key=lambda a: a[-1].real)
                left, right = cup_pair
                if 0.01 < right[-1].imag < 0.49:
                    cup_arc = PEArc(arctype='cup')
                    cup_arc.append(left[-1])
                    if left[-2].real > left[-1].real and right[-2].real < right[-1].real:
                        # / \ the cup wraps
                        cup_arc.append(PEPoint(0.0, left[-1].imag, leave_gap=True))
                        cup_arc.append(PEPoint(1.0, left[-1].imag))
                        cup_arc.append(right[-1])
                    else:
                        cup_arc.append(right[-1])
                    n = len(self.arcs)
                    self.arcs.append(cup_arc)
                    self.curve_graph.add_edge(self.arcs.index(left), n)
                    self.curve_graph.add_edge(n, self.arcs.index(right))
        # For manifolds which are not S^3 knot complements, closed curves in the
        # PE character variety can wrap vertically around the pillowcase.
        wrappers = [n for n, a in enumerate(self.arcs) if (
            a[0].index[0] == 0 and
            a[-1].index[0] == self.order - 1 and
            0.001 < a[0].real < 0.999 and
            0.001 < a[-1].real < 0.9999)]
        if wrappers:
            def f(n):
                a = self.arcs[n]
                if a[0].real <= 0.5:
                    # Glue front to back at the bottom.
                    # This is approximate because are 1 step from the start
                    A = array([abs(1.0 - b[-1].real - a[-1].real) for b in self.arcs])
                    result = int(A.argmin())
                    print(A[result])
                else:
                    # Glue back to front at the top.
                    # This is essentially exact.
                    A = array([abs(1.0 - b[0].real - a[0].real) for b in self.arcs])
                    result = int(A.argmin())
                    print(A[result])
                return result
            for n in wrappers:
                self.curve_graph.add_edge(n, f(n))

    def show(self, show_group=False):
        """Plot the pillowcase image of this PE Character Variety."""
        self.build_arcs(show_group)
        Plot(self.arcs,
             number_type=PEPoint,
             limits=((0.0, 1.0), (0.0, 0.5)),
             margins=(0, 0),
             aspect='equal',
             position=(0.07, 0.07, 0.8, 0.95),
             title='PE Character Variety of %s'%self.manifold.name(),
             colors=self.colors,
             extra_lines=[((0.5, 0.5), (0.0, 1.0))],
             extra_line_args={'color':'black', 'linewidth':0.75},
             show_group=show_group)

    def show_R_longitude_evs(self):
        """
        Display a plot of the longitude eigenvalues on the meridian
        preimage of the R-circle.
        """
        self.elevation.show_R_longitude_evs()

    def show_T_longitude_evs(self):
        """
        Display a plot of the longitude eigenvalues on the meridian
        preimage of the T-circle.
        """
        self.elevation.show_T_longitude_evs()

    def show_volumes(self):
        E = self.elevation
        volumes = E.compute_volumes(E.T_fibers)
        args = [angle(z) for z in E.T_circle]
        args = [float(a + 2*pi) if a <= 0 else a for a in args]
        data = [[v if v is None else complex(a, v) for a, v in zip(args, lift)]
                for lift in volumes]
        Plot(data)

    def get_rep(self, fiber_index, shape_index, precision=1000, tight=True):
        """
        Return a precise representation, computed to the specified binary
        precision, determined by the shape set with index
        *shape_index* in the fiber with index fiber_index over the
        T_circle. If the optional keyname argument *tight* is set to
        False, the R circle is used instead.

        If the repesentation is a PSL(2,R) rep, the return value from
        this rep will be an object of type PSL2RRepOf3ManifoldGRoup
        class.  Otherwise, the return value will be of type
        PSL2CRepOF3ManifoldGRoup.
        """
        if tight:
            target = U1Q(-fiber_index, self.order, precision=precision)
        else:
            target = self.elevation.R_circle[fiber_index]
        fibers = self.elevation.T_fibers if tight else self.elevation.R_fibers
        rough_shapes = fibers[fiber_index].shapes[shape_index]
        polished_shapes = PolishedShapeSet(rough_shapes, target, precision)
        rho = PSL2CRepOf3ManifoldGroup(self.manifold, target,
                                       rough_shapes, precision)
        if rho.is_PSL2R_rep():
            rho = PSL2RRepOf3ManifoldGroup(rho)
        rho.index = (shape_index, fiber_index)
        rho.rep_type = polished_shapes.rep_type()
        return rho
