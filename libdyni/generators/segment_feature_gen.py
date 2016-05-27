# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import joblib
from itertools import compress
from libdyni.features.segment_feature_extractor import *
from libdyni.utils.feature_container import FeatureContainer
from libdyni.utils.segment_container import SegmentContainer
from libdyni.features.audio_chunk_extractor import AudioChunkExtractor


logger = logging.getLogger(__name__)


class SegmentFeatureGen:

    def __init__(self, feature_extractors, **kwargs):

        # TODO use Python abc (Abstract Base Classes)?
        if not all(isinstance(fe, SegmentFeatureExtractor) for fe in feature_extractors):
            raise Exception("All feature extractors must be instances of SegmentFeatureExtractor.")
        self.__feature_extractors = feature_extractors

        if "ff_gen" in kwargs:
            self.__frame_feature_gen = kwargs.get("ff_gen")
        if "audio_root" in kwargs:
            self._audio_root = kwargs.get("audio_root")

    def execute(self, segment_container):
        """
        Args:
            - segment_container            
        Returns tuple (segment_container, segment_container_has_features)
        """

#        # build segment container path
#        segment_container_path = os.path.join(
#                self._segment_container_root,
#                os.path.splitext(os.path.basename(audio_path))[0] + ".sc.jl")
#        sc = SegmentContainer.load(segment_container_path)

        # check if segment container already has all wanted features
        has_features = segment_container.has_features([fe.name for fe in self.__feature_extractors])
        if all(has_features):
            logger.debug("Segment container has all requested features")
            return

        # get feature container if needed
        if any(isinstance(fe, SegmentFrameBasedFeatureExtractor) for fe in compress(self.__feature_extractors, [not hf for hf in has_features])):
            fc, created = self.__frame_feature_gen.execute(
                    (self._audio_root, segment_container.audio_path))
            if created:
                logger.debug("Feature container created")

        for fe in compress(self.__feature_extractors, [not hf for hf in has_features]):
            if isinstance(fe, AudioChunkExtractor):
                fe.execute(segment_container)
            elif isinstance(fe, SegmentFrameBasedFeatureExtractor):
                fe.execute(segment_container, fc)
            else:
                raise Exception("Segment feature extractor not implemented")

