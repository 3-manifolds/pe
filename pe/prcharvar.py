# -*- coding: utf-8 -*-
"""
Defines the main class PRCharVariety.

An PRCharVariety object represents a Peripherally Real Character Variety,
generically consisting of representations which send peripheral elements to real
loxodromic elements of SL(2,C).  Each PRCHarVariety object manages a
LineElevation object, which represent a family of fibers for the meridian
holonomy map lying above the interval (0, 1] on the real axis.
[[ Should the interval be [-1,1] with 0 omitted, so we have the same symmetry
as in the circle case? ]]
"""
from __future__ import print_function
from numpy import log
import numpy as np
from snappy import Manifold, ManifoldHP
from spherogram import Graph
from collections import OrderedDict
from .elevation import LineElevation
from .point import PEPoint
from .plot import Plot
import sys, os

def quad_fit(x, y):
    """
    Given 1-dim arrays x and y describing 6 points in the plane, find a
    quadratic almost passing through the middle two points and which
    least-squares approximates the others.
    """
    x, y = np.asarray(x), np.asarray(y)
    assert len(x) == len(y) == 6
    A = np.asarray([np.ones(6), x, x*x]).transpose()
    B, z = A.copy(), y.copy()
    # Modify the equations a bit so that we will pass very close to the middle points
    B[2:4] *= 10
    z[2:4] *= 10
    poly = np.linalg.lstsq(B, z)[0]
    #Ccompute actual error
    error = np.linalg.norm(np.dot(A, poly) - y)
    return poly, error

class PRCharVariety(object):
    """
    An object representating the PE Character Variety of a 3-manifold.
    """
    def __init__(self, manifold, order=128, offset=0.02, elevation=None,
                 base_dir='PR_base_fibers', hint_dir='PR_hints',
                 ignore_saved=True):
        self.base_dir = base_dir
        if not isinstance(manifold, (Manifold, ManifoldHP)):
            manifold = Manifold(manifold)
        self.offset = offset
        self.order = order
        if elevation is None:
            self.elevation = LineElevation(
                manifold,
                order=order,
                offset=offset,
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
        Find the arcs of this PR Character Variety.  Iterate through the arcs in
        T_longitude_evs, extracting the subarcs where the longitude eigenvalue
        lies on the real axis.
        """
        self.arcs = []
        elevation = self.elevation
        for m, track in enumerate(elevation.T_longitude_evs):
            arc = []
            marker = ''
            failed = elevation.tighten_failures.get(m, set())
            for n, ev in enumerate(track):
                if n in failed:
                    # Skip over this point since we weren't able to tighten it.
                    continue
                # Is the longitude eigenvalue non-zero and real?
                if  ev and abs(ev) > 1.0E-1000 and abs(ev.imag)/abs(ev.real) < 1.0E-6:
                    if show_group:
                        shape = elevation.T_fibers[n].shapes[m]
                        # Since the peripheral holonomy is hyperbolic,
                        # if these shapes give a PSL(2, R) repn then
                        # the shapes themselves must be flat.
                        if shape.has_real_shapes():
                            marker = '.'
                        else:
                            marker = 'x'
                    L = log(abs(ev.real))
                    # Take the square root, so we have an eigenvalue not a holonomy.
                    M = 0.5*log(elevation.T_path[n])
                    arc.append(PEPoint(L, M, marker=marker, index=(n, m)))
                else:
                    if len(arc) > 1:  # Ignore isolated parobolic reps.
                        self.arcs.append(arc)
                    # start a new arc
                    arc = []
            # If the last point on the track is a real rep, we end up here with
            # a non-empty arc.
            if len(arc) > 1:
                self.arcs.append(arc)
        self.add_extrema()

    def add_extrema(self):
        """
        Tries to improve the picture by adding caps/cups as
        appropriate.  The code is pretty basic since so far we have
        only seen components that look like hyperbolas where we have
        only to add a single cap joining two existing arcs.

        To avoid a flat plateau at the top of each hill, we compute
        the parabolic approximations of the ends of the two arcs we
        wish to join and use those to fill in the gap.
        """
        progress = True
        while progress:
            progress = False
            for i, arc in enumerate(self.arcs):
                for other in self.arcs[i+1:]:
                    # The last condition in the below test is to avoid
                    # joining asymptotes towards the same ideal point.
                    if arc[0].imag == other[0].imag and arc[0].imag > -1.5:
                        if abs(arc[0].real - other[0].real) < 1.5:
                            points = [arc[1], arc[2], arc[0], other[0], other[1], other[2]]
                            xs = [p.real for p in points]
                            ys = [p.imag for p in points]
                            poly, error = quad_fit(xs, ys)
                            if error < 1e-2:
                                xfill = np.linspace(xs[2], xs[3], 6)[1:-1]
                                yfill = poly[0] + poly[1]*xfill + poly[2]*xfill*xfill
                                new_seg = [PEPoint(x, y, marker='h') for x, y in zip(xfill, yfill)]
                                self.arcs.remove(other)
                                self.arcs[i] = list(reversed(arc)) + new_seg + other
                                progress = True
                            break
                if progress:
                    break

        self.curve_graph = curve_graph = Graph([], list(range(len(self.arcs))))
        # build the color dict
        self.colors = OrderedDict()
        for n, component in enumerate(curve_graph.components()):
            for m in component:
                self.colors[m] = n
    
    def show(self, show_group=False):
        """Plot this PR Character Variety."""
        self.build_arcs(show_group)
        self.plot = Plot(self.arcs,
             number_type=PEPoint,
             margins=(0, 0),
             position=(0.07, 0.07, 0.8, 0.8),
             title='PR Character Variety of %s'%self.manifold.name(),
             show_group=show_group)
