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

from dynibatch.features.extractors.frame_feature import AudioFrameFeatureExtractor


class EnergyExtractor(AudioFrameFeatureExtractor):
    """Computes the energy of the signal."""

    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        """Class method required by all feature extractors (even though some arguments
        are not used."""
        return cls()

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        """
            :return: size of the feature
        """
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        return np.sum(data**2)
