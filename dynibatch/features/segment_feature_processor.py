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
from itertools import compress

from dynibatch.features.extractors.segment_feature import SegmentFeatureExtractor
from dynibatch.features.extractors.segment_feature \
    import SegmentFrameBasedFeatureExtractor
from dynibatch.features.extractors.audio_chunk import AudioChunkExtractor
from dynibatch.features.extractors.chirplets_chunk import ChirpletsChunkExtractor

__all__ = ['SegmentFeatureProcessor']

logger = logging.getLogger(__name__)


class SegmentFeatureProcessor:
    # TODO (jul) change name to SegmentFeatureExtractor and replace current
    # SegmentFeatureExtractor by something else.
    """Class holding all objects needed to run segment-based feature extractors."""

    def __init__(self, feature_extractors, **kwargs):
        """Initializes segment feature processor.

        Args:
            feature_extractors (list of SegmentFeatureExtractor): list of feature
                extractors to be executed.
            **kwarg:
                ff_pro (FrameFeatureProcessor)
                audio_root (str): audio root path
        """

        # TODO use Python abc (Abstract Base Classes)?
        if not all(isinstance(fe, SegmentFeatureExtractor) for fe in feature_extractors):
            feature_extractors_types = ",".join([str(type(fe)) for fe in feature_extractors])
            raise TypeError(
                "All feature extractors must be instances of SegmentFeatureExtractor.\n" +
                "Types in the actual given list:\n" +
                feature_extractors_types)
        self._feature_extractors = feature_extractors

        if "ff_pro" in kwargs:
            self._frame_feature_pro = kwargs.get("ff_pro")
        if "audio_root" in kwargs:
            self._audio_root = kwargs.get("audio_root")

    def execute(self, segment_container):
        """
        Args:
            segment_container
        
        Fills the segment containers with the features specified in
            feature_extractors
        """

        logger.debug("Processing %s segment container",
                     segment_container.audio_path)

        # check if segment container already has all wanted features
        has_features = segment_container.has_features(
            [fe.name for fe in self._feature_extractors])
        if all(has_features):
            logger.debug("Segment container has all requested features")
            return

        # get feature container if needed
        if any(isinstance(fe,
                          SegmentFrameBasedFeatureExtractor) for fe in compress(
                              self._feature_extractors, [not hf for hf in has_features])):
            fc, created = self._frame_feature_pro.execute(
                (self._audio_root, segment_container.audio_path))
            if created:
                logger.debug("Feature container created")

        for fe in compress(self._feature_extractors,
                           [not hf for hf in has_features]):
            if isinstance(fe, AudioChunkExtractor):
                fe.execute(segment_container)
            elif isinstance(fe, SegmentFrameBasedFeatureExtractor):
                fe.execute(segment_container, fc)
            elif isinstance(fe, ChirpletsChunkExtractor):
                fe.execute(segment_container)
            else:
                raise TypeError(
                    "Segment feature extractor {} not implemented".format(fe.name))
