from libdyni.features.segment_feature_extractor import SegmentFrameBasedFeatureExtractor

class FrameFeatureChunkExtractor(SegmentFrameBasedFeatureExtractor):

    def __init__(self, name):
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name

    def execute(self, segment_container, feature_container):

        for s in segment_container.segments:
            
            start_ind = feature_container.time_to_frame_ind(s.start_time)
            end_ind =  start_ind + feature_container.time_to_frame_ind(s.duration)

            s.features[self._name] =  feature_container.features[self._name]["data"][start_ind:end_ind]
