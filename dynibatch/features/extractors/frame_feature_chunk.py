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

from dynibatch.features.extractors.segment_feature import SegmentFrameBasedFeatureExtractor


logger = logging.getLogger(__name__)


class FrameFeatureChunkExtractor(SegmentFrameBasedFeatureExtractor):
    """Extracts chunks of frame-based features."""

    def __init__(self, name, pca=None, scaler=None):
        """Initializes frame feature chunk extractor.

        Args:
            name (str): name of the frame-based feature.
            pca (sklearn.decomposition.PCA) (optional): Principal component analysis
                (PCA)
            scaler (sklearn.preprocessing.StandardScaler) (optional): standardize
                features by removing the mean and scaling to unit variance.

        Note: if both scaler and pca are set, the pca is performed first
        """

        super().__init__()
        self._name = name
        self._scaler = scaler
        self._pca = pca

    @property
    def name(self):
        return self._name

    def execute(self, segment_container, feature_container):
        """Gets chunk of features from a feature container and sets it to every
        segment in a segment container.

        Args:
            segment_container (SegmentContainer): segment container to set the
                chunk of data to.
            feature_container (FeatureContainer): feature container to get the
                chunk of data from.

        Sets the chunks of features to every segment in segment_container.
        """

        for seg in segment_container.segments:
            start_ind = feature_container.time_to_frame_ind(seg.start_time)
            end_ind = start_ind + feature_container.time_to_frame_ind(seg.duration)

            if end_ind > len(feature_container.features[self.name]["data"]):
                # that can happen if the end time of the latest analysis frame
                # is earlier than the end time of the segment
                logger.debug("Segment {0:.3f}-{1:.3f} from {2} end time".format(
                    seg.start_time,
                    seg.end_time,
                    segment_container.audio_path) +
                             " exceed feature container size for feature {}.".format(self.name))
                break

            data = feature_container.features[self.name]["data"][start_ind:end_ind]
            if self._pca:
                data = self._pca.transform(data)
            if self._scaler:
                data = self._scaler.transform(data)
            seg.features[self.name] = data
