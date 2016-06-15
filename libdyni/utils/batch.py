import logging
import numpy as np


logger = logging.getLogger(__name__)


def gen_minibatches(segment_container_gen,
        classes,
        batch_size,
        n_features,
        n_time_bins,
        feature_name,
        active_segment_only=False):
    """Generates batches of segments from segment container generator"""

    if n_features == 1:
        batch = np.zeros((batch_size, 1, n_time_bins), dtype=np.float32)
    else:
        batch = np.zeros((batch_size, 1, n_features, n_time_bins), dtype=np.float32)
    targets = np.empty((batch_size), dtype=np.int16)

    count = 0
    for sc in segment_container_gen.execute():
        logger.debug("iterate_minibatch: {}".format(sc.audio_path))
        for s in sc.segments:
            if not feature_name in s.features:
                break
            if not active_segment_only or s.activity:
                if n_features == 1:
                    batch[count, 0, :] = s.features[feature_name].T # TODO (jul) sort out shape
                else:
                    batch[count, 0, :, :] = s.features[feature_name].T # TODO (jul) sort out shape
                targets[count] = classes.index(s.label)
                count += 1
                if count == batch_size:
                    count = 0
                    yield batch, targets

