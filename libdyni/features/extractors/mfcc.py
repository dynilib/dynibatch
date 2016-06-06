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
        log_amp (boolean): whether or not to compute the log of the mel
        spectrum.
    """

    #TODO (jul) add top_db for logamplitude

    def __init__(self,
                 sample_rate=44100,
                 fft_size=512,
                 n_mels=128,
                 n_mfcc=32,
                 min_freq=0,
                 max_freq=22050):

        super().__init__()

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.min_freq = min_freq
        self.max_freq = max_freq

        self._mel_basis = librosa.filters.mel(sr=sample_rate,
                                              n_fft=fft_size,
                                              n_mels=n_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
        self._dct_basis = librosa.filters.dct(n_mfcc, n_mels)

    @property
    def name(self):
        """
        Returns:
            The name of SegmentFrameBasedFeatureExtractor, it is also its type
        """
        return 'mfcc'

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
        data = librosa.logamplitude(np.dot(self._mel_basis, data), top_db=None)
        return np.dot(self._dct_basis, data)
