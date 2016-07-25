import logging
import numpy as np


logger = logging.getLogger(__name__)


class MiniBatchGen:
    """Generates batches of segments from segment container generator"""

    def __init__(self,
            classes,
            feature_name,
            batch_size,
            n_features,
            n_time_bins):
 
        self.classes = classes
        self.feature_name = feature_name
        self.batch_size = batch_size
        self.n_features = n_features

        if n_features == 1:
            self.minibatch = np.zeros((batch_size, 1, n_time_bins), dtype=np.float32)
        else:
            self.minibatch = np.zeros((batch_size, 1, n_features, n_time_bins), dtype=np.float32)
        self.filenames = np.empty((batch_size), dtype="|U200")
        self.targets = np.empty((batch_size), dtype=np.int16)

    def execute(self,
            segment_container_gen,
            active_segments_only=False,
            with_targets=False,
            with_filenames=False):

        count = 0
        for sc in segment_container_gen.execute():
            logger.debug("iterate_minibatch: {}".format(sc.audio_path))
            for s in sc.segments:
                if not self.feature_name in s.features:
                    break
                if not active_segments_only or s.activity:
                    if self.n_features == 1:
                        self.minibatch[count, 0, :] = s.features[self.feature_name].T # TODO (jul) sort out shape
                    else:
                        self.minibatch[count, 0, :, :] = s.features[self.feature_name].T # TODO (jul) sort out shape
                    if with_filenames:
                        self.filenames[count] = sc.audio_path
                    if with_targets:
                        self.targets[count] = self.classes.index(s.label)
                    count += 1
                    if count == self.batch_size:
                        count = 0
                        data = [self.minibatch]
                        if with_targets:
                            data.append(self.targets)
                        if with_filenames:
                            data.append(self.filenames)
                        yield tuple(data)


