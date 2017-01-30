#The MIT License
#
#Copyright (c) 2017 DYNI machine learning & bioacoustics team - Univ. Toulon
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from enum import Enum
from scipy.signal import hann
from dynibatch.utils.exceptions import DynibatchError


class WindowType(Enum):
    rect = 0
    hanning = 1


def window(win_type, size):
    """Return a precalculated window with a defined size and type

        Args:
            win_type (WindowType): type of the window wanted
            size (int): size of the window
        Returns:
            a precalculated (win_type) window with (size) as size
    """

    if win_type == WindowType.hanning:
        # use asymetric window (https://en.wikipedia.org/wiki/Window_function#Symmetry)
        return hann(size, sym=False)
    else:
        raise DynibatchError("Window type {} is not defined".format(win_type))
