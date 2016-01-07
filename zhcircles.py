"""
How to make a plot:

sage: run zhcircles.py
sage: df = db.dataframe()
sage: F = make_plot(df.ix['m016'])
sage: F.save_tikz('m016.pdf')   # .pdf important, otherwise saves as png

Old to do's for the paper:

Q1. Why is there a missing component on m389?

A1. There is a component corresponding to representations factoring
through C_2 * C_3 = PSL(2, Z).


Q2: Issue with t11462: There's a "parabolic" point in the translation
extension variety that is not seen by ptolemy, namely (0, -2) or
symmetrically (1, 2).  


A2. This is because of a "Tillmann point", that is, a representation
that can't be realized for the given triangulation.

First, we tried randomizing the triangulation and computing with
ptolemy/magma from scratch always saw only the one parabolic
representation where the longitude has trace -2.  Let's look at this
example in more detail:

sage: rho2 = L.rep_dict[2,1]
sage: rho2
<t11462(0,0): [0.32474, 0.99977+0.0071319I, 0.99977-0.0071319I, 1.0001+0.0079763I, 1, 3.0807, 1.0001-0.0079763I, 1.7545]>
sage: RDF(rho2(l).trace())
1.9902037446379561
sage: rho2til(l)
<PSL2tilde: A = [[-4.33709,-7.06504],[4.02575,6.32730]]; s = -2>

Certainly, the shapes are degenerating, but does this correspond to an
ideal point or a Tillmann point?  For rho2, the max trace of a
generator is 8.2, slightly less than the corresponding number for the
known parabolic. Also, Regina claims this manifold is small, so there
can't be an ideal point associated to a closed surface. 

So I guess it's just a very robust Tillmann point. Indeed, the code in
"examples/t11462.py" uses Magma to confirm (starting with a
presentation for the fundamental group) that there is a representation
into SL(2, C) where all peripheral elements are parabolic with trace
+2, and this representation is definitely not found by ptolemy.

To do: 

Counterexample to: 

real place + simple root => (-infinity, 1)



Problems: 

m103: skips
m389: component of mult 2? 
o9_01035: skips
o9_02163: not symmetric
s068: skips
s114: skips
s301: skips
s385: vertical jump?
s936: vertical jump?
t00027: missing component?
t00565: not symmetric
t02069: not symmetric
t04557: not symmetric
t04697: very scattered
t08979: seriously no symmetric
v1964: not symmetric
v3505: just deadends

Interesing examples

t03608 crosses the L=0 line without any unimodular reducible reps.
t09612 this is pretty wierd.


Only 21 with unimodular roots of multiplicity > 1.  

t12247 <-- multiple root but not tangent to L=0!
o9_30426
o9_38639

Odd one:

o9_09530
t03608


Does this have an ideal point? 

t11462





For talk:

m016
m201 - red to parabolic on axis
m222 - mult. roots
m389 - deadend component from degree 2 component
s841 - parabolic to parabolic
v0220 - basic example with interleaving
t11462 - parabolic to parabolic only 
o9_34801 - don't get half + epsilon of everything. 
o9_04139 - everything
"""

import snappy
import taskdb2
import pe
import cPickle as pickle
import bz2
import md5
from sage.all import ComplexField, RealField, PolynomialRing, QQ, RR
from pe.plot import MatplotPlot as Plot
import nplot
import matplotlib.style
#matplotlib.style.use('classic')
import pandas as pd


def initial_database():
    names = []
    for M in snappy.OrientableCuspedCensus(cusps=1):
        M.dehn_fill((1,0))
        if M.homology().order() == 1:
            names.append(M.name())

    cols = [('volume', 'double'), ('tets', 'int'), ('inS3', 'tinyint'), 
            ('alex', 'text'), ('alex_deg', 'int'), ('alex_monic', 'tinyint'),
            ('num_uniroots', 'int'), ('num_mult_uniroots', 'int'),
            ('num_psl2R_arcs', 'int'), ('real_places', 'int'),
            ('parabolic_PSL2R', 'int'), ('base_index', 'int'),
            ('radius', 'double'), ('transa_rcs', 'mediumblob')
    ]
    db = taskdb2.ExampleDatabase('ZHCircles', names, cols)
    db.new_task_table('task0')
    return db

def basic(task):
    M = snappy.Manifold(task['name'])
    task['volume'] = M.volume()
    task['tets'] = M.num_tetrahedra()
    M.dehn_fill((1,0))
    if M.fundamental_group().num_generators() == 0:
        task['inS3'] = True
    else:
        task['inS3'] = False
    task['done'] = True

def on_unit_circle(z):
    prec = z.parent().prec()
    return abs(abs(z) - 1) < 2.0**(-0.8*prec)
    
def unimodular_roots(p):
    CC = ComplexField(212)
    return [(z, e) for z, e in p.roots(CC) if on_unit_circle(z)]

def num_real_roots(p):
    return len(p.roots(RR))

def rotation_angle(z):
    twopi = 2*z.parent().pi()
    arg = z.argument()/twopi
    if arg < 0:
        arg += 1
    return arg
    
def alexander(task):
    M = snappy.Manifold(task['name'])
    p = M.alexander_polynomial()
    task['alex'] = repr(p)
    task['alex_deg'] = p.degree()
    task['alex_monic'] = p.is_monic()
    uniroots = unimodular_roots(p)
    task['num_uniroots'] = len(uniroots)
    task['num_mult_uniroots'] = len([z for z, e in uniroots if e > 1])
    task['done'] = True

def parabolic_psl2R(task):
    R = PolynomialRing(QQ, 'x')
    M = snappy.Manifold(task['name'])
    assert M.homology().elementary_divisors() == [0]
    ans = 0
    obs_classes = M.ptolemy_obstruction_classes()
    assert len(obs_classes) == 2
    for obs in obs_classes:
        V = M.ptolemy_variety(N=2, obstruction_class=obs)
        for sol in V.retrieve_solutions():
            p = R(sol.number_field())
            if p == 0:  # Field is Q
                n = 1
            else:
                n = num_real_roots(p)
            ans += n
            try:
                if sol.is_geometric():
                    task['real_places'] = n
            except:
                if obs._index > 0:
                    task['real_places'] = n
            
                            
    task['parabolic_PSL2R'] = ans
    if 'real_places' in task:
        task['done'] = True
    return task

def longitude_translation(rho):
    a, b = rho.manifold.homological_longitude()
    G = rho.polished_holonomy(1000)
    rhotil = rho.lift_on_cusped_manifold()
    x, y = rhotil.peripheral_translations()
    ans = abs(a*x + b*y)
    assert abs(ans - ans.round()) < 1e-100
    return ans.round()
    
def real_parabolic_reps_from_ptolemy(M, pari_prec=15):
    R = PolynomialRing(QQ, 'x')
    RR = RealField(1000)
    ans = []
    obs_classes = M.ptolemy_obstruction_classes()
    for obs in obs_classes:
        V = M.ptolemy_variety(N=2, obstruction_class=obs)
        #for sol in V.retrieve_solutions():
        for sol in V.compute_solutions(engine='magma'):
            prec = snappy.pari.set_real_precision(pari_prec)
            is_geometric = sol.is_geometric()
            snappy.pari.set_real_precision(prec)
            cross_ratios = sol.cross_ratios()
            shapes = [R(cross_ratios['z_0000_%d' % i].lift())
                      for i in range(M.num_tetrahedra())]
            s = shapes[0]
            p = R(sol.number_field())
            if p == 0:  # Field is Q
                assert False
            # print p, '\n'
            for r in p.roots(RR, False):
                rho = pe.real_reps.PSL2RRepOf3ManifoldGroup(
                    M, target_meridian_holonomy = RR(1),
                    rough_shapes=[cr(r) for cr in shapes])
                rho.is_galois_conj_of_geom = is_geometric
                ans.append(rho)
    return ans
                    

def parabolic_psl2R_details(task, pari_prec=15):
    M = snappy.Manifold(task['name'])
    reps = real_parabolic_reps_from_ptolemy(M, pari_prec)
    ans = []
    for rho in reps:
        ans.append( (longitude_translation(rho), rho.is_galois_conj_of_geom) )
    ans.sort()
    task['parabolic_PSL2R_details']  = repr(ans)
    task['done'] = True
    return task


def comp_pickle(obj):
    return bz2.compress(pickle.dumps(obj, 2), 9)

def unpickle_comp(data):
    return pickle.loads(bz2.decompress(data))

def save_plot_data(task):
    V, L = None, None
    for r in [1.02, 1.04, 1.08, 1.12, 1.20]:
        try:
            V = pe.PECharVariety(task['name'], radius=r)
            L = pe.SL2RLifter(V)
            break
        except ValueError:
            continue

    if L is not None:
        task['base_index'] = L.elevation.base_index
        task['radius'] = L.elevation.radius
        task['num_psl2R_arcs'] = len(L.translation_arcs)
        task['trans_arcs'] = comp_pickle(L._show_homological_data())
        task['done'] = True
    return task

def save_plot_highres(task):
    V = pe.PECharVariety(task['name'], radius=task['radius'], order=2048)
    L = pe.SL2RLifter(V)
    task['trans_arcs_highres'] = comp_pickle(L._show_homological_data())
    task['done'] = True
    
def make_plots(df=None):
    if df is None:
        db = taskdb2.ExampleDatabase('ZHCircles')
        df = db.dataframe()
    df = df[(df.trans_arcs.notnull())&(df.parabolic_PSL2R_details.notnull())]
    for i, row in df.iterrows():
        if row['num_psl2R_arcs'] == 0:
            continue
        plot = make_plot(row)
        plot.save('/tmp/plots/' + row['name'] + '.pdf')


class PaperPlot(Plot):
    @staticmethod
    def color(i):
        return 'black'

plot_default = {
    'longitude line':'green',
    'simple alex root':'red',
    'mult alex root':'purple',
    'galois of geom':'orange',
    'other parabolic':'white',
    'plot_cls':Plot
}

plot_paper = {
    'longitude line':'black',
    'simple alex root':'grey',
    'mult alex root':'black',
    'galois of geom':'black',
    'other parabolic':'0.9',
    'plot_cls':PaperPlot
}



def make_plot(row, params=plot_default):
    if row['trans_arcs_highres'] is not None:
        arcs = row['trans_arcs_highres'].unpickle()
    else:
        arcs = row['trans_arcs'].unpickle()
    title = '$' + row['name'] + '$: genus = ' + repr(row['alex_deg']//2)
    plot = params['plot_cls'](arcs, title=title)
    ax = plot.figure.axis
    ax.plot((0, 1), (0, 0), color=params['longitude line'])
    ax.legend_.remove()
    ax.set_xbound(0, 1)

    alex = PolynomialRing(QQ, 'a')(row['alex'])
    for z, e in unimodular_roots(alex):
        theta = rotation_angle(z)
        if e == 1:
            color = params['simple alex root']
        else:
            color = params['mult alex root']
        ax.plot([theta], [0], color=color, marker='o',
                ms=7, markeredgecolor=color)

    M = snappy.Manifold(row['name'])
    for L, is_galois_conj_of_geom in eval(row['parabolic_PSL2R_details']):
        if is_galois_conj_of_geom:
            color = params['galois of geom']
        else:
            color = params['other parabolic']
        for x, y in [(0, L), (0, -L), (1, L), (1, -L)]:
            ax.plot([x], [y], color=color, marker='o', ms=10,
                    markeredgecolor='black')
            
    plot.figure.draw()
    return plot


class ZHCircles(taskdb2.ExampleDatabase):
    def __init__(self):
        taskdb2.ExampleDatabase.__init__(self, 'ZHCircles')

    def dataframe(self):
        ans = taskdb2.ExampleDatabase.dataframe(self)
        return ans.set_index('name', drop=False)


if __name__ == '__main__':
    #initial_database()
    #db = ZHCircles()
    #df = db.dataframe()
    #task = db.get_row(199)
    #print db.remaining('task0')
    #db.run_function('task0', save_plot_data, num_tasks=1)
    df = pd.read_pickle('zhcircle.pickle')
    for name in ['m016', 'v0220', 'm389', 'm201',
                 'm222', 's841', 't11462', 'o9_34801', 'o9_04139']:
        F = make_plot(df.ix[name], plot_paper)
        F.save_tikz(name + '.pdf')
