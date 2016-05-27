import numpy as np
import librosa

from libdyni.features.frame_feature_extractor import PowerSpectrumFrameFeatureExtractor


class MelSpectrumExtractor(PowerSpectrumFrameFeatureExtractor):
    """Computes the mel spectrum.
    
    Attribute:
        sample_rate (int)
        fft_size (int)
        n_mels(int)
        min_freq (int)
        max_freq (int)
        log_amp (boolean): whether or not to compute the log of the mel
        spectrum.    
    """

    #TODO (jul) add top_db for logamplitude

    def __init__(self, sample_rate=44100, fft_size=512,
            n_mels=128, min_freq=0, max_freq=22050, log_amp=True):

        super().__init__()

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.log_amp = log_amp
        self._mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=fft_size,
                n_mels=n_mels,
                fmin=min_freq,
                fmax=max_freq)

    @property
    def name(self):
        return 'mel_spectrum'

    @property
    def size(self):
        return self.n_mels
    
    @property
    def config(self):
        return {'sample_rate': self.sample_rate,
                'fft_size': self.fft_size,
                'n_mels': self.n_mels,
                'min_freq': self.min_freq,
                'max_freq': self.max_freq,
                'log_amp': self.log_amp}

    def execute(self, data):
        """Computes the mel spectrum.

        Args:
            data (numpy array): power spectrum

        Returns the mel spectrum as a numpy array
        """

        if self.log_amp:
            return librosa.logamplitude(
                    np.dot(self._mel_basis, data),
                    top_db=None)
        return np.dot(self._mel_basis, data)
