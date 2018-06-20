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
import os
import random
import time
import joblib

from dynibatch.utils.segment import CommonLabels
from dynibatch.utils.stats import get_stats
from dynibatch.utils.exceptions import ParameterError, DynibatchError


logger = logging.getLogger(__name__)


"""
A datasplit is dict containing an id and three datasets (train. valid, test),
each represented by a set of audio file paths. Audio file paths must be given
relatively to some common audio root path.

It allows to keep track and share a dataset split to replicate some experiments.

Example:

    somedatasplit = {
                        "id": "someid",
                        "sets":
                            {
                                "train": set([
                                    "some/path/file1.wav",
                                    "some/path/file3.wav",
                                    "some/path/file4.wav",
                                    "some/path/file10.wav"]),
                                "valid": set([
                                    "some/path/file2.wav",
                                    "some/path/file9.wav"]),
                                "test": set([
                                    "some/path/file5.wav",
                                    "some/path/file6.wav",
                                    "some/path/file7.wav",
                                    "some/path/file8.wav"])
                            }
                    }

"""


def create_datasplit_default(dataset, name=None):
    """Create a datasplit dict from a single user-defined data set

    Args:
        dataset (set): set of audio file paths
        name (str): name of the datasplit (set to unix timestamp if not
            specified)

    Returns:
        a dict with a set of audio file paths
    """
    if not name:
        name = int(time.time())
    return {"id": "{}".format(name),
            "sets": {"default": dataset}
           }


def create_datasplit(train_set, validation_set, test_set, name=None):
    """Create a datasplit dict from user-defined data sets

    Args:
        train_set (set): set of audio file paths
        valid_set (set): set of audio file paths
        test_set (set): set of audio file paths
        name (str): name of the datasplit (set to unix timestamp if not
            specified)

    Returns:
        a dict with train_set, validation_set and test_set, as sets of
        audio file paths
    """
    if not name:
        name = int(time.time())
    return {"id": "{}".format(name),
            "sets": {"train": train_set,
                     "validation": validation_set,
                     "test": test_set}
           }


def create_random_datasplit(segment_containers,
                            train_ratio=0.65,
                            validation_ratio=0.0,
                            test_ratio=0.35):
    """Create a stratified data split. Only allowed for datasets having a single
    label per file.

    Args:
        segment_containers: list of SegmentContainer instances,
        train_ratio: ratio of files in training set
        validation_ratio: ratio of files in validation set
        test_ratio: ratio of files in test set

    Returns:
        a dict with train_set, validation_set and test_set, as sets of
        audio file paths
    """

    if train_ratio + validation_ratio + test_ratio != 1:
        raise ParameterError(
            "train_ratio + validation_ratio + test_ratio must be equal to 1")

    train_set = set()
    validation_set = set()
    test_set = set()

    # get class set
    classes = set()
    for sc in segment_containers:
        classes |= sc.labels

    # remove segment.CommonLabels
    classes.discard(CommonLabels.garbage.value)
    classes.discard(CommonLabels.no_activity.value)
    classes.discard(CommonLabels.unknown.value)

    # for every label, get audio_path set and split
    for label in classes:

        file_set = {sc.audio_path for sc in segment_containers if
                    label in sc.labels}

        num_files = len(file_set)

        train_file_subset_size = int(round(num_files * train_ratio))
        validation_file_subset_size = int(round(num_files * validation_ratio))

        # create train_set from random subset of size train_subset_size
        train_file_subset = set(random.sample(file_set, train_file_subset_size))
        
        # fix possible size error due to rounding
        if validation_file_subset_size > len(file_set - train_file_subset):
            validation_file_subset_size = len(file_set - train_file_subset)

        # then create validation_set from remaining files
        validation_file_subset = set(random.sample(file_set - train_file_subset,
                                                   validation_file_subset_size))

        # then create test_set from remaining files
        test_file_subset = file_set - train_file_subset - validation_file_subset

        if len(train_file_subset) < 2:
            logger.warning("The number of files in the train set for label %s "
                           "is smaller than 2", label)
        if validation_ratio > 0 and len(validation_file_subset) < 2:
            logger.warning(
                "The number of files in the validation set for label %s "
                "is smaller than 2", label)
        if len(test_file_subset) < 2:
            logger.warning("The number of files in the test set for label %s "
                           "is smaller than 2", label)

        train_set |= train_file_subset
        test_set |= test_file_subset
        validation_set |= validation_file_subset

    # make sure no file is in two sets
    if len({sc.audio_path for sc in segment_containers}) < \
    len(train_set | validation_set | test_set):
        logger.warning("Some files are in several sets")

    return create_datasplit(train_set, validation_set, test_set)


def write_datasplit(datasplit, path, compress=0):
    """Writes datasplit to joblib pickle

    Args:
        datasplit (dict): datasplit dict, as defined in the description of this
            module.
        path (str): write path
        compress (int from 0 to 9): compression level

    Writes:
        A joblib dump.
    """

    joblib.dump(datasplit,
                os.path.join(path, "datasplit_{}.jl".format(datasplit["id"])),
                compress=compress)


def get_datasplit_stats(segment_containers, datasplit):
    """Compute basic statistics on dataset.

    Args:
        segment_containers (SegmentContainer iterator): segment containers
            containing the data
        datasplit (dict): datasplit dict, as defined in the description of this
            module.

    Returns:
        A string containing basic statistics on the dataset.
    """

    train_set = [sc for sc in segment_containers if
                 sc.audio_path in datasplit["sets"]['train']]
    validation_set = [sc for sc in segment_containers if
                      sc.audio_path in datasplit["sets"]['validation']]
    test_set = [sc for sc in segment_containers if
                sc.audio_path in datasplit["sets"]['test']]

    if not train_set:
        raise DynibatchError("No train set")
    train_stats = get_stats(train_set)
    if validation_set:
        validation_stats = get_stats(validation_set)
    if test_set:
        test_stats = get_stats(test_set)

    classes = sorted(list(set(l for sc in train_set for l in sc.labels)))

    column_width = [10, 15, 23, 16, 24]

    string_of_stats = "Sets statistics (training/validation/test)\n"
    string_of_stats += "{0}{1}{2}{3}{4}\n".format(
        "Class".center(column_width[0]),
        "Num files".center(column_width[1]),
        "Num active files".center(column_width[2]),
        "Num segments".center(column_width[3]),
        "Num active segments".center(
            column_width[4]))

    for label in classes:
        string_of_stats += "{0}{1}{2}{3}{4}\n".format(
            label.center(column_width[0]),
            "{0}/{1}/{2}" \
            .format(
                train_stats['per_class'][label]['num_files'] if train_set else 0,
                validation_stats['per_class'][label]['num_files'] if validation_set else 0,
                test_stats['per_class'][label]['num_files'] if test_set else 0) \
            .center(column_width[1]),
            "{0}/{1}/{2}" \
            .format(
                train_stats['per_class'][label]['num_active_files'] if train_set else 0,
                validation_stats['per_class'][label]['num_active_files'] if validation_set else 0,
                test_stats['per_class'][label]['num_active_files'] if test_set else 0) \
            .center(column_width[2]),
            "{0}/{1}/{2}" \
            .format(
                train_stats['per_class'][label]['num_segments'] if train_set else 0,
                validation_stats['per_class'][label]['num_segments'] if validation_set else 0,
                test_stats['per_class'][label]['num_segments'] if test_set else 0) \
            .center(column_width[3]),
            "{0}/{1}/{2}" \
            .format(
                train_stats['per_class'][label]['num_active_segments'] if train_set else 0,
                validation_stats['per_class'][label]['num_active_segments'] \
                if validation_set else 0,
                test_stats['per_class'][label]['num_active_segments'] if test_set else 0) \
            .center(column_width[4]))

    return string_of_stats
