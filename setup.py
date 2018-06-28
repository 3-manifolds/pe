from setuptools import setup

setup( name = 'pe',
       version = '0.1.1',
       install_requires = ['snappy>=2.7a1'],
       dependency_links = [],
       packages = ['pe'],
       package_dir = {'pe' : 'pe'},
       zip_safe = False,
       description= 'Peripherally (parabolic) and elliptic reprs of 3-manifold groups', 
       author = 'Marc Culler and Nathan M. Dunfield',
       author_email = 'marc.culler@gmail.com, nathan@dunfield.info',
       license='GPLv2+',
       url = 'https://bitbucket.org/t3m/pe',
       classifiers = [
           'Development Status :: 4 - Beta',
           'Intended Audience :: Science/Research',
           'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
           'Operating System :: OS Independent',
           'Programming Language :: Python',
           'Topic :: Scientific/Engineering :: Mathematics',
        ],
)
