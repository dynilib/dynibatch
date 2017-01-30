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


# -*- coding: utf-8 -*-

import logging
import soundfile as sf
from dynibatch.utils.exceptions import DynibatchError

from dynibatch.utils.windows import WindowType, window


logger = logging.getLogger(__name__)


class AudioFrameGen:
    """Audio frame generator.
    
    Generates windowed audio frames.
    """

    def __init__(self, sample_rate, win_size, hop_size, win_type=WindowType.hanning):
        """Initializes audio frame generator.

        Args:
            sample rate (int): sample rate in Hz
            win_size (int): frame size in samples
            hop_size (int): hop size in samples
        """

        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._win_type = win_type

        # Create window
        if win_type != WindowType.rect:
            self._window = window(win_type, win_size)

    @property
    def config(self):
        return {"win_size": self._win_size,
                "hop_size": self._hop_size,
                "win_type": self._win_type.value}

    def execute(self, path):
        """Executes the frame generator.
        
        Args:
            path (str): path of the audio file

        Yields:
            Audio frames as numpy arrays
        """

        audio, sr = sf.read(path)

        # make sure the file is mono
        if len(audio.shape) != 1:
            raise DynibatchError("Please use only mono files")

        # make sure the actual sample rate is the same as specified in the init
        if sr != self._sample_rate:
            raise DynibatchError("Sample rate mismatch in file {}: ".format(path) +
                                 "{} instead of {}.".format(sr, self._sample_rate))

        i = 0
        while i + self._win_size < audio.shape[0]:
            if self._win_type != WindowType.rect:
                yield self._window * audio[i:i+self._win_size]
            else:
                yield audio[i:i+self._win_size]
            i += self._hop_size
