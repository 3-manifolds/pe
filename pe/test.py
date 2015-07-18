import getopt, sys
from . import sage_helper
from . import complex_reps, real_reps, shape

modules = [complex_reps, real_reps, shape]


if __name__ == '__main__':
    optlist, args = getopt.getopt(sys.argv[1:], 'v', ['verbose'])
    verbose = len(optlist) > 0
    sage_helper.doctest_modules(modules, verbose)
