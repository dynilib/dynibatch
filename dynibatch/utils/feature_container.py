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


import os
from collections import defaultdict
import joblib
from dynibatch.utils import exceptions




FC_EXTENSION = ".fc.jl"


class FeatureContainer:
    """Feature container.

    A feature container is related to an audio file. It contains all the
    short-term features (e.g. spectral flatness, mel-spectra...), as well as their
    parameters, computed from this audio file. When saved on disk, features
    are not directly stored in segments because that would imply a lot of duplicated
    data (since segments most often overlap). Instead, they are saved as feature
    container dumps.
    """

    def __init__(self, audio_path, sample_rate, win_size, hop_size):
        """Initializes feature container.

        Args:
            audio_path (str): audio path (relative to some audio root path)
            sample rate (int): sample rate in Hz
            win_size (int): size of the audio frame, in samples
            hop_size (int): audio frames hop size, in samples
        """

        self._audio_path = audio_path
        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self.features = defaultdict(dict)

    def has_features(self, features):
        """Check whether the feature container has a set of features with a
        given config.

        Args:
            features: list of tuple (name, config)
        Returns a list of booleans
        """

        return [name in self.features and
                self.features.get(name).get("data") is not None and
                config == self.features.get(name).get("config", dict())
                for name, config in features]

    @staticmethod
    def load(path):
        """Loads a feature container from a joblib dump.

        Args:
            path (str)
        """

        try:
            fc = joblib.load(path)
            if not isinstance(fc, FeatureContainer):
                raise exceptions.ParsingError(
                    "Object in {} is not an instance of FeatureContainer".format(
                        path))
            return fc
        except FileNotFoundError:
            return None

    def save(self, path, compress=0):
        """Save a feature container as a joblib dump.

        Args:
            path (str)
            compress (int between 0 and 9): compression level
        """

        filename =  os.path.join(path, os.path.splitext(self._audio_path)[0] + FC_EXTENSION)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(self, filename, compress=compress)

    def time_to_frame_ind(self, time):
        """Computes the frame index from a time
        
        Args:
            time (float): time in second
        """

        return int(time * self._sample_rate / self._hop_size)
