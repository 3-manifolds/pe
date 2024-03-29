from setuptools import setup
import re

version_data = open('pe/version.py').read()
version = re.match("version = '(.*)'", version_data).groups()[0]

setup( name = 'pe',
       version = version,
       install_requires = ['snappy>=2.6.1'],
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
