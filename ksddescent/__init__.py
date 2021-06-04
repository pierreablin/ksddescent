# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
__version__ = '0.1dev'

from .ksd_descent import ksdd_gradient, ksdd_lbfgs  # noqa
from .contenders import svgd, mmd_lbfgs  # noqa
