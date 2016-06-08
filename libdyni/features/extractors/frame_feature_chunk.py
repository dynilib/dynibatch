from libdyni.features.extractors.segment_feature import SegmentFrameBasedFeatureExtractor

class FrameFeatureChunkExtractor(SegmentFrameBasedFeatureExtractor):
    """Extracts chunks of frame-based features.

    Attributes:
        name (str): name of the frame-based feature.
        scaler (sklearn.preprocessing.StandardScaler) (optional): standardize
        features by removing the mean and scaling to unit variance.

    """
    def __init__(self, name, scaler=None):
        super().__init__()
        self.name = name
        self.scaler = scaler

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
                raise ValueError("Segment {0:.3f}-{1:.3f} from {2} end time" +
                        " exceed feature container size.".format(s.start_time,
                                s.end_time, segment_container.audio_path))

            if self.scaler:
                s.features[self.name] = self.scaler.transform(feature_container.features[self.name][
                    "data"][start_ind:end_ind])
            else:
                s.features[self.name] = feature_container.features[self.name][
                    "data"][start_ind:end_ind]
