import numpy as np
import librosa

from libdyni.features.extractors.frame_feature \
    import PowerSpectrumFrameFeatureExtractor


class MFCCExtractor(PowerSpectrumFrameFeatureExtractor):
    """Computes the MFCC.

    Attribute:
        sample_rate (int)
        fft_size (int)
        n_mels(int)
        n_mfcc (int)
        min_freq (int)
        max_freq (int)
        top_db (float): threshold log amplitude at top_db below the peak:
            max(log(S)) - top_db
    """

    def __init__(self,
                 sample_rate=44100,
                 fft_size=512,
                 n_mels=128,
                 n_mfcc=32,
                 min_freq=0,
                 max_freq=22050,
                 top_db=None):

        super().__init__()

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.top_db = top_db

        self._mel_basis = librosa.filters.mel(sr=sample_rate,
                                              n_fft=fft_size,
                                              n_mels=n_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
        self._dct_basis = librosa.filters.dct(n_mfcc, n_mels)

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        return self.n_mfcc

    @property
    def config(self):
        return {'sample_rate': self.sample_rate,
                'fft_size': self.fft_size,
                'n_mels': self.n_mels,
                'n_mfcc': self.n_mfcc,
                'min_freq': self.min_freq,
                'max_freq': self.max_freq}

    def execute(self, data):
        """Computes the MFCC.

        Args:
            data (numpy array): power spectrum

        Returns the mfcc as a numpy array
        """
        data = librosa.logamplitude(np.dot(self._mel_basis, data),
                                    top_db=self.top_db)
        return np.dot(self._dct_basis, data)
