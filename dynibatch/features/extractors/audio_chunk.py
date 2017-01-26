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


import logging
from os.path import join
from soundfile import SoundFile
import numpy as np
from dynibatch.features.extractors.segment_feature import SegmentFeatureExtractor


logger = logging.getLogger(__name__)


class AudioChunkExtractor(SegmentFeatureExtractor):
    """Extracts the audio chunk corresponding to every segment in a segment
    container."""

    def __init__(self, audio_root, sample_rate):
        """Initializes audio chunk extractor.

        Args:
            audio_root (str): audio files root path
            sample_rate (int): sample rate of all audio files in audio_root (they
            must all have the same sample rate)
        """
        super().__init__()
        self._audio_root = audio_root
        self._sample_rate = sample_rate

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    def execute(self, segment_container):
        """Executes the audio chunk extractor.

        Args:
            segment_container (SegmentContainer)
        """

        with SoundFile(join(self._audio_root,
                            segment_container.audio_path)) as f:

            if not f.seekable():
                raise ValueError("file must be seekable")

            # make sure the actual sample rate is the same as specified in the init
            if f.samplerate != self._sample_rate:
                raise Exception("Sample rate mismatch in file {}: ".format(path) +
                        "{} instead of {}.".format(sf.info(path).samplerate,
                            self._sample_rate))

            for s in segment_container.segments:

                start_time = s.start_time
                end_time = s.end_time

                n_samples = int(np.rint(
                    (end_time - start_time) * self._sample_rate))

                start_ind = int(start_time * self._sample_rate)

                if start_ind + n_samples > len(f):
                    # TODO (jul) use specific exception or, maybe better, just remove it
                    # because an exception might already be raised in f.read below
                    raise ValueError(
                        "Segments {0}-{1} exceeds file {2} duration".format(
                            start_time, end_time, segment_container.audio_path))

                f.seek(start_ind)
                s.features[self.name] = f.read(n_samples, dtype="float32")
