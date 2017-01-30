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


from sys import float_info
import numpy as np
from dynibatch.features.extractors.frame_feature import SpectrumFrameFeatureExtractor


class SpectralFlatnessExtractor(SpectrumFrameFeatureExtractor):
    """Spectral flatness extractor.

    The spectral flatness is defined by the ratio between the geometric mean and the arithmetic
    mean.
    """

    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        """Class method required by all frame-feature extractors (even though some arguments
        are not used."""
        return cls()

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        """Computes the spectral flatness.

        Args:
            data (numpy array)

        Returns:
            float: spectral flatness
        """

        data = np.where(data == 0, float_info.epsilon, data) # replace 0s by epsilon to avoid log(0)
        sum_mag_bins = np.sum(data)
        return np.exp(np.sum(np.log(data)) / len(data)) / (sum_mag_bins / len(data))
