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


import numpy as np

from dynibatch.features.extractors.frame_feature import SpectrumFrameFeatureExtractor


class FFTSpectrumExtractor(SpectrumFrameFeatureExtractor):
    """Computes the FFT spectrum."""

    def __init__(self, fft_size):
        """
        Args:
            fft_size (int)
        """

        super().__init__()

        self._fft_size = fft_size

    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        """Class method required by all frame-feature extractors (even though some arguments
        are not used."""
        return cls(
            fft_size=audio_frame_config["win_size"])

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        return int(self._fft_size / 2 + 1)

    @property
    def config(self):
        return {'fft_size': self._fft_size}

    def execute(self, data):
        """Computes the fft spectrum.

        Args:
            data (numpy array): spectrum

        Returns:
            the spectrum (class created only for feature extraction
            code consistency).
        """

        return data
