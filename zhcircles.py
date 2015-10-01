import snappy
import taskdb2
import pe
import cPickle as pickle
import bz2
import md5
from sage.all import ComplexField, PolynomialRing, QQ, RR
from pe.plot import MatplotPlot as Plot


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
            ('radius', 'double'), ('trans_arcs', 'mediumblob')
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

def make_plots():
    db = taskdb2.ExampleDatabase('ZHCircles')
    df = db.dataframe()
    df = df[df.trans_arcs.notnull()]
    for i, d in df.iterrows():
        if d['num_psl2R_arcs'] == 0:
            continue
        arcs = d['trans_arcs'].unpickle()
        plot = Plot(arcs, title=d['name'] + ' reframed')
        ax = plot.figure.axis
        ax.plot((0, 1), (0, 0), color='green')
        ax.legend_.remove()
        ax.set_xbound(0, 1)
        plot.figure.draw()
        plot.save('plots/' + d['name'] + '.pdf')


class ZHCircles(taskdb2.ExampleDatabase):
    def __init__(self):
        taskdb2.ExampleDatabase.__init__(self, 'ZHCircles')


if __name__ == '__main__':
    #initial_database()
    db = ZHCircles()
    #df = db.dataframe()
    #task = db.get_row(199)
    #print db.remaining('task0')
    #db.run_function('task0', save_plot_data, num_tasks=1)
