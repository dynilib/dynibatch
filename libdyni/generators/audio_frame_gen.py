# -*- coding: utf-8 -*-

import logging
from enum import Enum
from scipy.signal import hann
import soundfile as sf


logger = logging.getLogger(__name__)


class Window(Enum):
    rect = 0
    hanning = 1


class AudioFrameGen:
    """Audio frame generator."""

    def __init__(self, sample_rate, win_size, hop_size, win_type=Window.hanning):

        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._win_type = win_type

        if not isinstance(win_type, Window):
            raise TypeError("win_type must be an instance of Window")

        # Create window
        if self._win_type == Window.hanning:
            # use asymetric window (https://en.wikipedia.org/wiki/Window_function#Symmetry)
            self._window = hann(self._win_size, sym=False)
        elif self._win_type == Window.rect:
            pass  # no windowing

    @property
    def config(self):
        return {"win_size": self._win_size,
                "hop_size": self._hop_size,
                "win_type": self._win_type.value}

    def execute(self, path):
        """ Yields the windowed frames"""

        # make sure the file is mono
        if sf.info(path).channels != 1:
            raise Exception("Please use only mono files")
        
        # make sure the actual sample rate is the same as specified in the init
        if sf.info(path).samplerate != self._sample_rate:
            raise Exception("Sample rate mismatch in file {}: ".format(path) +
                    "{} instead of {}.".format(sf.info(path).samplerate,
                        self._sample_rate))

        for frame in sf.blocks(path,
                               blocksize=self._win_size,
                               overlap=self._win_size-self._hop_size,
                               dtype="float32"):
            if len(frame) < self._win_size:
                break # last frame
            if self._win_type != Window.rect:
                yield self._window * frame
            else:
                yield frame
