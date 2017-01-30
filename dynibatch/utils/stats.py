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


import os
import numpy as np
import soundfile as sf


def get_file_durations(path):
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                filepath = os.path.join(dirpath, filename)
                sample_rate = sf.info(filepath).samplerate
                # the number of frame is not returned by sf.info
                frames = sf.SoundFile(filepath)._info.frames
                yield float(frames) / sample_rate
            except IOError:
                pass  # not an audio file


def get_stats(segment_containers):
    """
    Returns the following segments statistics:
        - number of classes
        - number of segments / class: min, max, mean, histogram
        - number of active segments / class: min, max, mean, histogram
        - number of active files / class: min, max, mean, histogram
    """

    stats = {}

    # per-class stats

    stats['per_class'] = {}

    classes = sorted(list(set(
        label for sc in segment_containers for label in sc.labels)))
    for label in classes:

        stats_per_class = {}

        sc_subset = [sc for sc in segment_containers if label in sc.labels]
        sc_active_subset = [sc for sc in sc_subset if sc.n_active_segments > 0]

        stats_per_class['num_segments'] = sum(
            sc.n_segments_with_label(label) for sc in sc_subset)
        stats_per_class['num_active_segments'] = sum(
            sc.n_active_segments_with_label(label) for sc in sc_active_subset)
        stats_per_class['num_files'] = len(sc_subset)
        stats_per_class['num_active_files'] = len(sc_active_subset)

        stats['per_class'][label] = stats_per_class

    # global stats

    stats['classes'] = classes

    num_segments = np.asarray(
        [stats['per_class'][c]['num_segments'] for c in classes])
    num_segments_dict = {}
    num_segments_dict['list'] = num_segments
    num_segments_dict['min'] = np.min(num_segments)
    num_segments_dict['max'] = np.max(num_segments)
    num_segments_dict['mean'] = np.mean(num_segments)
    stats['num_segments'] = num_segments_dict

    num_active_segments = np.asarray(
        [stats['per_class'][c]['num_active_segments'] for c in classes])
    num_active_segments_dict = {}
    num_active_segments_dict['list'] = num_active_segments
    num_active_segments_dict['min'] = np.min(num_active_segments)
    num_active_segments_dict['max'] = np.max(num_active_segments)
    num_active_segments_dict['mean'] = np.mean(num_active_segments)
    stats['num_active_segments'] = num_active_segments_dict

    num_files = np.asarray(
        [stats['per_class'][c]['num_files'] for c in classes])
    num_files_dict = {}
    num_files_dict['list'] = num_files
    num_files_dict['min'] = np.min(num_files)
    num_files_dict['max'] = np.max(num_files)
    num_files_dict['mean'] = np.mean(num_files)
    stats['num_files'] = num_files_dict

    num_active_files = np.asarray(
        [stats['per_class'][c]['num_active_files'] for c in classes])
    num_active_files_dict = {}
    num_active_files_dict['list'] = num_active_files
    num_active_files_dict['min'] = np.min(num_active_files)
    num_active_files_dict['max'] = np.max(num_active_files)
    num_active_files_dict['mean'] = np.mean(num_active_files)
    stats['num_active_files'] = num_active_files_dict

    return stats
