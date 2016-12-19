import logging

from libdyni.features.extractors.segment_feature import SegmentFrameBasedFeatureExtractor


logger = logging.getLogger(__name__)


class FrameFeatureChunkExtractor(SegmentFrameBasedFeatureExtractor):
    """Extracts chunks of frame-based features.

    Attributes:
        name (str): name of the frame-based feature.
        pca (sklearn.decomposition.PCA) (optional): Principal component analysis
        (PCA)
        scaler (sklearn.preprocessing.StandardScaler) (optional): standardize
            features by removing the mean and scaling to unit variance.
    Note: if both scaler and pca are set, the pca is performed first

    """
    def __init__(self, name, pca=None, scaler=None):
        super().__init__()
        self.name = name
        self.scaler = scaler
        self.pca = pca

    def execute(self, segment_container, feature_container):
        """Gets chunk of features from a feature container and sets it to every
        segments in a segment container.

        Args:
            segment_container (SegmentContainer): segment container to set the
            chunk of data to.
            feature_container (FeatureContainer): feature container to get the
            chunk of data from.

        Sets the chunks of features to every segment in segment_container.
        """

        for s in segment_container.segments:
            start_ind = feature_container.time_to_frame_ind(s.start_time)
            end_ind = start_ind + feature_container.time_to_frame_ind(s.duration)

            if end_ind > len(feature_container.features[self.name]["data"]):
                # that can happen if the end time of the latest analysis frame
                # is earlier than the end time of the segment
                logger.debug("Segment {0:.3f}-{1:.3f} from {2} end time".format(
                    s.start_time,
                    s.end_time,
                    segment_container.audio_path) +
                             " exceed feature container size for feature {}.".format(self.name))
                break

            data = feature_container.features[self.name]["data"][start_ind:end_ind]
            if self.pca:
                data = self.pca.transform(data)
            if self.scaler:
                data = self.scaler.transform(data)
            s.features[self.name] = data
