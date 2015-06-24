from . import shapes, closed, polish_reps, real_reps
import doctest

for module in [shapes, closed, polish_reps, real_reps]:
    print(module.__name__ + ': ' + repr(doctest.testmod(module)))
