# -*- coding: utf-8 -*-

import logging
from scipy.signal import hann
import soundfile as sf

from libdyni.utils.windows import WindowType, window


logger = logging.getLogger(__name__)


class AudioFrameGen:
    """Audio frame generator."""

    def __init__(self, sample_rate, win_size, hop_size, win_type=WindowType.hanning):

        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._win_type = win_type

        # Create window
        if not win_type == WindowType.rect:
            self._window = window(win_type, win_size)

    @property
    def config(self):
        return {"win_size": self._win_size,
                "hop_size": self._hop_size,
                "win_type": self._win_type.value}

    def execute(self, path):
        """ Yields the windowed frames"""

        audio, sr = sf.read(path)

        # make sure the file is mono
        if len(audio.shape) != 1:
            raise Exception("Please use only mono files")
        
        # make sure the actual sample rate is the same as specified in the init
        if sr != self._sample_rate:
            raise Exception("Sample rate mismatch in file {}: ".format(path) +
                    "{} instead of {}.".format(sr, self._sample_rate))

        i = 0
        while i + self._win_size < audio.shape[0]:
            if self._win_type != WindowType.rect:
                yield self._window * audio[i:i+self._win_size]
            else:
                yield audio[i:i+self._win_size]
            i += self._hop_size
