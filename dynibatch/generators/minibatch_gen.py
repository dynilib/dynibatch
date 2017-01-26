import logging
import json
import numpy as np
import joblib

# generators
from dynibatch.generators.audio_frame_gen import AudioFrameGen
from dynibatch.generators.segment_container_gen import SegmentContainerGenerator
# features
from dynibatch.features.frame_feature_processor import FrameFeatureProcessor
from dynibatch.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from dynibatch.features.segment_feature_processor import SegmentFeatureProcessor
from dynibatch.features import extractors
# activity detection
from dynibatch.features import activity_detection
# utils
from dynibatch.parsers import label_parsers
from dynibatch.utils.segment import CommonLabels


logger = logging.getLogger(__name__)


class MiniBatchGen:
    """Generates minibatches of features.
    
    Features are pulled from segments in the SegmentContainer objects yielded by
    a SegmentContainerGenerator.
    """

    def __init__(self,
                 segment_container_gen,
                 feature_name,
                 batch_size,
                 n_features,
                 n_time_bins):
        """Initializes minibatch generator.

        Args:
            segment_container_gen (SegmentContainerGenerator)
            feature_name (str): name of the feature to pull from segments
            batch_size (int): minibatch size in number of segments
            n_features (int): number of features, as returned by the size property
                of the feature extractor
            n_time_bins (int): number of time bins of the feature in a segment    
        """

        self._segment_container_gen = segment_container_gen
        self._feature_name = feature_name
        self._batch_size = batch_size
        self._n_features = n_features
        self._n_time_bins = n_time_bins

    @classmethod
    def from_json_config_file(cls, config_path):
        """Creates a dict of minibatch generators from a config file

        One minibatch generator is created for every set defined in the
        datasplit, if specified in the config file. Otherwise, only one is
        created.
        
        Args:
            config_path: path to the config file (a json file)
        Returns:
            A dict with one item per set defined in the datasplit, such as
            out_dict["<set_name>"] = <minibatch generator for this set>. If no
            datasplit is defined, the set is named "default".
        """

        # list of frame feature names and config required to compute segment
        # based features
        frame_feature_config_list = []

        # parse json file
        with open(config_path) as config_file:
            config = json.loads(config_file.read())

        # audio frame config
        af_config = config["audio_frame_config"]

        # segment config
        seg_config = config["segment_config"]

        # minibatch config
        batch_size = config["batch_size"]
        num_frames_per_seg = int(seg_config["seg_duration"] *
                                 af_config["sample_rate"] / af_config["hop_size"])

        # Create a label parser.
        # Because FileLabelParser is set with a file path and SegmentLabelParser
        # with a root path, two different keys are used
        if "file2label_filename" in config:
            label_parser = label_parsers.CSVFileLabelParser(config["file2label_filename"],
                    label_file=config.get("label_file"))
        elif "seg2label_root" in config:
            label_parser = label_parsers.CSVSegmentLabelParser(config["seg2label_root"],
                    config["label_file"])
        else:
            label_parser = None

        # get activity detection
        if "activity_detection" in config:
            act_det_config = config["activity_detection"]
            act_det = activity_detection.factory(
                act_det_config["name"],
                audio_frame_config=af_config,
                feature_config=act_det_config.get("config"))
            # get features required by the activity detection
            frame_feature_config_list += act_det.frame_feature_config
        else:
            act_det = None

        # get feature that will feed the minibatch
        frame_feature_config_list.append(config["feature"])

        # instanciate all frame feature extractors
        # TODO check for redundancy
        frame_feature_extractors = []
        for ff_cfg in frame_feature_config_list:
            frame_feature_extractors.append(extractors.factory(
                ff_cfg["name"],
                audio_frame_config=af_config,
                feature_config=ff_cfg.get("config")))

        # create a frame feature processor, in charge of computing all short-term features
        ff_pro = FrameFeatureProcessor(
            AudioFrameGen(
                          sample_rate=af_config["sample_rate"],
                          win_size=af_config["win_size"],
                          hop_size=af_config["hop_size"]),
            frame_feature_extractors,
            feature_container_root=config.get('features_root')
        )

        # create needed segment-based feature extractors
        ffc_ext = FrameFeatureChunkExtractor(config['feature']['name'])

        # create a segment feature processor, in charge of computing all segment-based features
        sf_pro = SegmentFeatureProcessor(
            [act_det, ffc_ext] if act_det else [ffc_ext],
            ff_pro=ff_pro,
            audio_root=config["audio_root"])

        datasplit_path = config.get("datasplit_path")
        sc_gen_dict = {}
        if not datasplit_path:
            # if no datasplit is present in the config file,
            # create one segment container generator
            sc_gen_dict["default"] = SegmentContainerGenerator(
                config["audio_root"],
                sf_pro,
                label_parser=label_parser,
                seg_duration=seg_config["seg_duration"],
                seg_overlap=seg_config["seg_overlap"],
                randomize=config["randomize_batch"])
        else:
            # else create one per set in the datasplit
            datasplit = joblib.load(datasplit_path)
            for set_name, _ in datasplit["sets"].items():
                sc_gen_dict[set_name] = SegmentContainerGenerator(
                    config["audio_root"],
                    sf_pro,
                    label_parser=label_parser,
                    dataset=datasplit["sets"][set_name],
                    seg_duration=seg_config["seg_duration"],
                    seg_overlap=seg_config["seg_overlap"],
                    randomize=config["randomize_batch"])

        mb_gen_dict = {}
        for set_name, sc_gen in sc_gen_dict.items():
            mb_gen_dict[set_name] = MiniBatchGen(
                sc_gen,
                config['feature']['name'],
                batch_size,
                frame_feature_extractors[-1].size, # it is the last added
                num_frames_per_seg)

        return mb_gen_dict


    def start(self):
        """ start MiniBatchGen for generating minibatches """
        self._segment_container_gen.start()

    def reset(self):
        """ reset MiniBatchGen for regenerating minibatches """
        self._segment_container_gen.reset()

    def execute(self,
                active_segments_only=False,
                known_labels_only=False,
                with_targets=False,
                with_filenames=False):
        """Executes the minibatch generator

        Args:
            active_segments_only (bool): returns only segments with "activity" attribute set to True
            known_labels_only (bool): returns only segments with label not set to segment.CommonLabels.unknown.value
            with_targets (bool): returns labels associated to the data
            with_filenames (bool): returns filenames where the data were taken
        Returns:
            tuple(data, targets (if with_targets), filenames (if with_filenames))
        """

        if self._n_features == 1:
            minibatch = np.empty((self._batch_size, 1, self._n_time_bins),
                                 dtype=np.float32)
        else:
            minibatch = np.empty((self._batch_size, 1, self._n_features, self._n_time_bins),
                                 dtype=np.float32)

        if with_filenames:
            filenames = np.empty((self._batch_size), dtype="|U200")
        if with_targets:
            targets = np.empty((self._batch_size), dtype=np.int16)

        count = 0
        for sc in self._segment_container_gen.execute():
            logger.debug("iterate_minibatch: %s", sc.audio_path)
            for s in sc.segments:
                if self._feature_name not in s.features:
                    break
                if ((not active_segments_only or (hasattr(s, 'activity')) and
                    s.activity) and (not known_labels_only or s.label !=
                        CommonLabels.unknown.value)):
                    if self._n_features == 1:
                        minibatch[count, 0, :] = s.features[self._feature_name].T
                    else:
                        minibatch[count, 0, :, :] = s.features[self._feature_name].T
                    if with_filenames:
                        filenames[count] = sc.audio_path
                    if with_targets:
                        targets[count] = s.label

                    count += 1
                    if count == self._batch_size:
                        count = 0
                        data = [minibatch]

                        # create new arrays (alternatively, arrays could be copied when yielded)
                        if self._n_features == 1:
                            minibatch = np.empty((self._batch_size, 1, self._n_time_bins),
                                                 dtype=np.float32)
                        else:
                            minibatch = np.empty((self._batch_size, 1, self._n_features, self._n_time_bins),
                                                 dtype=np.float32)

                        if with_targets:
                            data.append(targets)
                            targets = np.empty((self._batch_size), dtype=np.int16)
                        if with_filenames:
                            data.append(filenames)
                            filenames = np.empty((self._batch_size), dtype="|U200")
                        yield tuple(data)
