# -*- coding: utf-8 -*-

import logging
from enum import Enum
from scipy.signal import hann
import soundfile as sf


LOGGER = logging.getLogger(__name__)


class Window(Enum):
    rect = 0
    hanning = 1


class AudioFrameGen:
    """Audio frame generator."""

    def __init__(self, win_size, hop_size, win_type=Window.hanning):

        self._win_size = win_size
        self._hop_size = hop_size
        self._win_type = win_type

        if not isinstance(win_type, Window):
            raise TypeError("win_type must be an instance of Window")

        # Create window
        if self._win_type == Window.hanning:
            # use asymetric window (https://en.wikipedia.org/wiki/Window_function#Symmetry)
            self._window = hann(self.win_size, sym=False)
        elif self._win_type == Window.rect:
            pass  # no windowing

    @property
    def win_size(self):
        return self._win_size

    @property
    def hop_size(self):
        return self._hop_size

    @property
    def win_type(self):
        return self._win_type

    @property
    def config(self):
        return {"win_size": self._win_size,
                "hop_size": self._hop_size,
                "win_type": self._win_type.value}

    def execute(self, path):
        """ Yields the windowed frames"""

        # make sure the file is mono
        if not sf.info(path).channels == 1:
            raise Exception("Please use only mono files")

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
