import numpy as np
import librosa
from libdyni.features.frame_feature_extractor import PowerSpectrumFrameFeatureExtractor


class MelSpectrumExtractor(PowerSpectrumFrameFeatureExtractor):

    def __init__(self, sample_rate=44100, fft_size=512,
            n_mels=128, min_freq=0, max_freq=22050):

        super().__init__()

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._n_mels = n_mels
        self._min_freq = min_freq
        self._max_freq = max_freq

        self.__mel_basis = librosa.filters.mel(sample_rate, fft_size)

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
    def min_freq(self):
        return self._min_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def name(self):
        return "mel_spectrum"

    @property
    def size(self):
        return self._n_mels
    
    @property
    def config(self):
        return {"sample_rate": self._sample_rate,
                "fft_size": self._fft_size,
                "n_mels": self._n_mels,
                "min_freq": self._min_freq,
                "max_freq": self._max_freq}

    def execute(self, data):
        """
        Args:
            data: power spectrum """

        return np.dot(self.__mel_basis, data)
