import logging
import os
import random
import time
import joblib

from libdyni.utils.segment import CommonLabels
from libdyni.utils.stats import get_stats
from libdyni.utils.exceptions import ParameterError


logger = logging.getLogger(__name__)


def create_datasplit(train_set, validation_set, test_set, name=None):
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
    """
    Stratified shuffle split.
    Splits the dataset in training, validation and testing data sets.
    The split is done for every classes except those in segment.
    Args:
        segment_containers: list of Segment instances,
        train_ratio: ratio of files in training set
        validation_ratio: ratio of files in validation set
        test_ratio: ratio of files in test set
    Returns a dictionary with train_set, validation_set and test_set, as lists of audio_path
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
    classes.discard(CommonLabels.garbage)
    classes.discard(CommonLabels.no_activity)
    classes.discard(CommonLabels.unknown)

    # for every label, get audio_path set and split
    for c in classes:

        file_set = {sc.audio_path for sc in segment_containers if
                    c in sc.labels}

        num_files = len(file_set)

        train_file_subset_size = int(round(num_files * train_ratio))
        validation_file_subset_size = int(round(num_files * validation_ratio))

        # create train_set from random subset of size train_subset_size
        train_file_subset = set(random.sample(file_set, train_file_subset_size))

        # then create validation_set from remaining files
        validation_file_subset = set(random.sample(file_set - train_file_subset,
                                                   validation_file_subset_size))

        # then create test_set from remaining files
        test_file_subset = file_set - train_file_subset - validation_file_subset

        if len(train_file_subset) < 2:
            logger.warning("The number of files in the train set for label %s is smaller than 2", c)
        if validation_ratio > 0 and len(validation_file_subset) < 2:
            logger.warning(
                "The number of files in the validation set for label %s is smaller than 2", c)
        if len(test_file_subset) < 2:
            logger.warning("The number of files in the test set for label %s is smaller than 2", c)

        train_set |= train_file_subset
        test_set |= test_file_subset
        validation_set |= validation_file_subset

    # make sure no file is in two sets
    if len({sc.audio_path for sc in segment_containers}) < \
    len(train_set | validation_set | test_set):
        logger.warning("Some files are in several sets")

    return create_datasplit(train_set, validation_set, test_set)

def write_datasplit(datasplit, path, compress=0):
    """
    Writes datasplit to joblib pickles.
    Args:
        datasplit: dictionary including the train, validation and test sets as list of file
        path: path in which to write the file
        compress: compression coefficient
    """

    joblib.dump(datasplit,
                os.path.join(path, "datasplit_{}.jl".format(datasplit["id"])),
                compress=compress)


def get_datasplit_stats(segment_containers, datasplit):
    """
    Get string describing basic statistics on dataset
    """

    train_set = [sc for sc in segment_containers if
                 sc.audio_path in datasplit["sets"]['train']]
    validation_set = [sc for sc in segment_containers if
                      sc.audio_path in datasplit["sets"]['validation']]
    test_set = [sc for sc in segment_containers if
                sc.audio_path in datasplit["sets"]['test']]

    if not train_set:
        raise Exception("No train set")
    train_stats = get_stats(train_set)
    if validation_set:
        validation_stats = get_stats(validation_set)
    if test_set:
        test_stats = get_stats(test_set)

    classes = sorted(list(set(l for sc in train_set for l in sc.labels)))

    column_width = [10, 15, 23, 16, 24]

    s = "Sets statistics (training/validation/test)\n"
    s += "{0}{1}{2}{3}{4}\n".format("Class".center(column_width[0]),
                                    "Num files".center(column_width[1]),
                                    "Num active files".center(column_width[2]),
                                    "Num segments".center(column_width[3]),
                                    "Num active segments".center(
                                        column_width[4]))
    for c in classes:
        s += "{0}{1}{2}{3}{4}\n".format(
            c.center(column_width[0]),
            "{0}/{1}/{2}".format(
                train_stats['per_class'][c]['num_files'] if train_set else 0,
                validation_stats['per_class'][c]['num_files'] if validation_set else 0,
                test_stats['per_class'][c]['num_files'] if test_set else 0).center(column_width[1]),
            "{0}/{1}/{2}".format(
                train_stats['per_class'][c]['num_active_files'] if train_set else 0,
                validation_stats['per_class'][c]['num_active_files'] if validation_set else 0,
                test_stats['per_class'][c]['num_active_files'] if test_set else 0).center(column_width[2]),
            "{0}/{1}/{2}".format(
                train_stats['per_class'][c]['num_segments'] if train_set else 0,
                validation_stats['per_class'][c]['num_segments'] if validation_set else 0,
                test_stats['per_class'][c]['num_segments'] if test_set else 0).center(column_width[3]),
            "{0}/{1}/{2}".format(
                train_stats['per_class'][c]['num_active_segments'] if train_set else 0,
                validation_stats['per_class'][c]['num_active_segments'] if validation_set else 0,
                test_stats['per_class'][c]['num_active_segments'] if test_set else 0).center(column_width[4]))

    return s
