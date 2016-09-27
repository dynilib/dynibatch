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
        self.n_time_bins = n_time_bins

    def execute(self,
            segment_container_gen,
            active_segments_only=False,
            with_targets=False,
            with_filenames=False):
        
        minibatch = np.empty((self.batch_size, self.n_features, self.n_time_bins), dtype=np.float32)
        
        if with_filenames:
            filenames = np.empty((self.batch_size), dtype="|U200")
        if with_targets:
            targets = np.empty((self.batch_size), dtype=np.int16)

        count = 0
        for sc in segment_container_gen.execute():
            logger.debug("iterate_minibatch: {}".format(sc.audio_path))
            for s in sc.segments:
                if not self.feature_name in s.features:
                    break
                if not active_segments_only or (hasattr(s, 'activity') and s.activity):
                    minibatch[count, :, :] = s.features[self.feature_name].T
                    if with_filenames:
                        filenames[count] = sc.audio_path
                    if with_targets:
                        targets[count] = self.classes.index(s.label)

                    count += 1
                    if count == self.batch_size:
                        count = 0
                        data = [minibatch]

                        minibatch = np.empty((self.batch_size, self.n_features, self.n_time_bins), dtype=np.float32)

                        if with_targets:
                            data.append(targets)
                            targets = np.empty((self.batch_size), dtype=np.int16)
                        if with_filenames:
                            data.append(filenames)
                            filenames = np.empty((self.batch_size), dtype="|U200")
                        yield tuple(data)


