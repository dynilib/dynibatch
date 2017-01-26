from enum import Enum
from scipy.signal import hann
from dynibatch.utils.exceptions import DynibatchError


class WindowType(Enum):
    rect = 0
    hanning = 1


def window(win_type, size):

    if win_type == WindowType.hanning:
        # use asymetric window (https://en.wikipedia.org/wiki/Window_function#Symmetry)
        return hann(size, sym=False)
    else:
        raise DynibatchError("Window type {} is not defined".format(win_type))
