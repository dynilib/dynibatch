import os
import numpy as np
import soundfile as sf


def get_file_durations(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                filepath = os.path.join(dirpath, filename)
                sample_rate = sf.info(filepath).samplerate
                frames = sf.SoundFile(filepath)._info.frames # the number of frame is not returned by sf.info
                yield float(frames) / sample_rate
            except:
                pass # not an audio file


def get_dataset_stats(segment_containers):
    """
    Returns the following segments statistics:
        - number of classes
        - segment duration: min, max, mean, variance, histogram
        - number of segments / class: min, max, mean, variance, histogram
        - number of files / class: min, max, mean, variance, histogram
        - total segments duration / class: min, max, mean, variance, histogram
    """

    stats = {}

    # per-class stats

    stats['per_class'] = {}

    classes = sorted(list(set(label for sc in segment_containers for label in sc.labels)))
    for c in classes:

        stats_per_class = {}

        sc_subset = [sc for sc in segment_containers if c in sc.labels]
        sc_active_subset = [sc for sc in sc_subset if sc.n_active_segments > 0]

        stats_per_class['num_segments'] = sum(sc.n_segments_with_label(c) for sc in sc_subset)
        stats_per_class['num_active_segments'] = sum(sc.n_active_segments_with_label(c) for sc in sc_active_subset)
        stats_per_class['num_files'] = len(sc_subset)

        stats['per_class'][c] = stats_per_class

    # global stats    

    stats['classes'] = classes

    num_segments = np.asarray( [ stats['per_class'][c]['num_segments'] for c in classes ])
    num_segments_dict = {}
    num_segments_dict['list'] = num_segments
    num_segments_dict['min'] = np.min(num_segments)
    num_segments_dict['max'] = np.max(num_segments)
    num_segments_dict['mean'] = np.mean(num_segments)
    stats['num_segments'] = num_segments_dict
    
    num_active_segments = np.asarray( [ stats['per_class'][c]['num_active_segments'] for c in classes ])
    num_active_segments_dict = {}
    num_active_segments_dict['list'] = num_active_segments
    num_active_segments_dict['min'] = np.min(num_active_segments)
    num_active_segments_dict['max'] = np.max(num_active_segments)
    num_active_segments_dict['mean'] = np.mean(num_active_segments)
    stats['num_active_segments'] = num_active_segments_dict

    num_files = np.asarray( [ stats['per_class'][c]['num_files'] for c in classes ])
    num_files_dict = {}
    num_files_dict['list'] = num_files
    num_files_dict['min'] = np.min(num_files)
    num_files_dict['max'] = np.max(num_files)
    num_files_dict['mean'] = np.mean(num_files)
    num_files_dict['var'] = np.var(num_files)
    stats['num_files'] = num_files_dict
    
    return stats

