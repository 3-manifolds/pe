# -*- coding: utf-8 -*-
"""
Defines the main class RLCharVariety.

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
from snappy import Manifold, ManifoldHP
from spherogram import Graph
from collections import OrderedDict
from .elevation import LineElevation
from .point import PEPoint
from .plot import Plot
import sys, os

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
                if  ev and abs(ev) > 1.0E-1000 and abs(ev.imag) < 1.0E-6:
                    if show_group:
                        shape = elevation.T_fibers[n].shapes[m]
                        if shape.has_real_traces():
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
        self.curve_graph = curve_graph = Graph([], list(range(len(self.arcs))))
        # build the color dict
        self.colors = OrderedDict()
        for n, component in enumerate(curve_graph.components()):
            for m in component:
                self.colors[m] = n
    
    def show(self, show_group=False):
        """Plot this PR Character Variety."""
        self.build_arcs(show_group)
        Plot(self.arcs,
             number_type=PEPoint,
             margins=(0, 0),
             position=(0.07, 0.07, 0.8, 0.8),
             title='PR Character Variety of %s'%self.manifold.name(),
             show_group=show_group)
