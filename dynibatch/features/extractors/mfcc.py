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
import librosa

from dynibatch.features.extractors.frame_feature \
    import PowerSpectrumFrameFeatureExtractor


class MFCCExtractor(PowerSpectrumFrameFeatureExtractor):
    """Computes the MFCC."""

    def __init__(self,
                 sample_rate=44100,
                 fft_size=512,
                 n_mels=128,
                 n_mfcc=32,
                 min_freq=0,
                 max_freq=22050,
                 top_db=None):
        """Initializes MFCC filters

        Args:
            sample_rate (int)
            fft_size (int)
            n_mels(int)
            n_mfcc (int)
            min_freq (int)
            max_freq (int)
            top_db (float): threshold log amplitude at top_db below the peak:
                max(log(S)) - top_db
        """


        super().__init__()

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._n_mels = n_mels
        self._n_mfcc = n_mfcc
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._top_db = top_db

        self._mel_basis = librosa.filters.mel(sr=sample_rate,
                                              n_fft=fft_size,
                                              n_mels=n_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
        self._dct_basis = librosa.filters.dct(n_mfcc, n_mels)
    
    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        """Class method required by all frame-feature extractors (even though some arguments
        are not used."""

        return cls(
                sample_rate=audio_frame_config["sample_rate"],
                fft_size=audio_frame_config["win_size"],
                n_mels=feature_config["n_mels"],
                n_mfcc=feature_config["n_mfcc"],
                min_freq=feature_config["min_freq"],
                max_freq=feature_config["max_freq"])

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        return self._n_mfcc

    @property
    def config(self):
        return {'sample_rate': self._sample_rate,
                'fft_size': self._fft_size,
                'n_mels': self._n_mels,
                'n_mfcc': self._n_mfcc,
                'min_freq': self._min_freq,
                'max_freq': self._max_freq}

    def execute(self, data):
        """Computes the MFCC.

        Args:
            data (numpy array): power spectrum

        Returns:
            the mfcc as a numpy array
        """
        data = librosa.logamplitude(np.dot(self._mel_basis, data),
                                    top_db=self._top_db)
        return np.dot(self._dct_basis, data)
