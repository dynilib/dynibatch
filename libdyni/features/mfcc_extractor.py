import numpy as np
import librosa
from libdyni.features.frame_feature_extractor import PowerSpectrumFrameFeatureExtractor


class MFCCExtractor(PowerSpectrumFrameFeatureExtractor):

    def __init__(self, sample_rate=44100, fft_size=512,
            n_mels=128, n_mfcc=32, min_freq=0, max_freq=22050):

        super().__init__()

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._n_mels = n_mels
        self._n_mfcc = n_mfcc
        self._min_freq = min_freq
        self._max_freq = max_freq

        self.__mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mels)
        self.__dct_basis = librosa.filters.dct(n_mfcc, n_mels)

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def fft_size(self):
        return self._fft_size

    @property
    def n_mels(self):
        return self._n_mels

    @property
    def n_mfcc(self):
        return self._n_mfcc

    @property
    def min_freq(self):
        return self._min_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def name(self):
        return "mfcc"

    @property
    def size(self):
        return self._n_mfcc
    
    @property
    def config(self):
        return {"sample_rate": self._sample_rate,
                "fft_size": self._fft_size,
                "n_mels": self._n_mels,
                "n_mfcc": self._n_mfcc,
                "min_freq": self._min_freq,
                "max_freq": self._max_freq}


    def execute(self, data, is_mel_spectrum=False):
        """Args:
                data: power spectrum or mel-spectrum
        """
        if not is_mel_spectrum:
            data = librosa.logamplitude(np.dot(self.__mel_basis, data), top_db=None)
        return np.dot(self.__dct_basis, data)
