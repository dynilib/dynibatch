import logging
import json
import numpy as np
import joblib

# generators
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
# features
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features import extractors
# activity detection
from libdyni.features import activity_detection
# utils
from libdyni.parsers import label_parsers
from libdyni.utils import exceptions


logger = logging.getLogger(__name__)


class MiniBatchGen:
    """Generates batches of segments from segment container generator"""

    def __init__(self,
                 segment_container_gen,
                 feature_name,
                 batch_size,
                 n_features,
                 n_time_bins):

        self.segment_container_gen = segment_container_gen
        self.feature_name = feature_name
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_time_bins = n_time_bins

    @classmethod
    def from_json_config_file(cls, config_path):
        """
            Create a dict of minibatch generators from JSON file
            (one per set if a datasplit is present in the config file)
            Args:
                config_path: path to the JSON config file
            Returns a minibatch generator
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
        
        # create a parser to get the labels from the label file
        label_parser = label_parsers.CSVLabelParser(config["label_file_path"])

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
            AudioFrameGen(win_size=af_config["win_size"],
                hop_size=af_config["hop_size"]),
            frame_feature_extractors,
            feature_container_root=config.get('features_root')
        )

        # create needed segment-based feature extractors
        ffc_ext = FrameFeatureChunkExtractor(config['feature']['name'])

        # create a segment feature processor, in charge of computing all segment-based features
        sf_pro = SegmentFeatureProcessor(
            [act_det, ffc_ext],
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
                seg_overlap=seg_config["seg_overlap"])
        else:
            # else create one per set in the datasplit
            datasplit = joblib.load(datasplit_path)
            for set_name, file_list in datasplit["sets"].items():
                sc_gen_dict[set_name] = SegmentContainerGenerator(
                    config["audio_root"],
                    sf_pro,
                    label_parser=label_parser,
                    dataset=datasplit["sets"][set_name],
                    seg_duration=seg_config["seg_duration"],
                    seg_overlap=seg_config["seg_overlap"])

        mb_gen_dict = {}
        for set_name, sc_gen in sc_gen_dict.items():
            mb_gen_dict[set_name] = MiniBatchGen(sc_gen,
                              config['feature']['name'],
                              batch_size,
                              frame_feature_extractors[-1].size, # it is the last added
                              num_frames_per_seg)

        return mb_gen_dict


    def start(self):
        """ start MiniBatchGen for generating minibatches """
        self.segment_container_gen.start()

    def reset(self):
        """ reset MiniBatchGen for regenerating minibatches """
        self.segment_container_gen.reset()

    def execute(self,
                active_segments_only=False,
                with_targets=False,
                with_filenames=False):
        """
            Produce a minibatch generator

            Args:
                active_segments_only: data returned only contain activities
                with_targets: return labels associated to the data
                with_filenames: return filenames where the data were taken
            Return: data + targets (if with_targets) + filenames (if with_filenames)
        """

        if self.n_features == 1:
            minibatch = np.empty((self.batch_size, 1, self.n_time_bins),
                                 dtype=np.float32)
        else:
            minibatch = np.empty((self.batch_size, 1, self.n_features, self.n_time_bins),
                                 dtype=np.float32)

        if with_filenames:
            filenames = np.empty((self.batch_size), dtype="|U200")
        if with_targets:
            targets = np.empty((self.batch_size), dtype=np.int16)

        count = 0
        for sc in self.segment_container_gen.execute():
            logger.debug("iterate_minibatch: %s", sc.audio_path)
            for s in sc.segments:
                if self.feature_name not in s.features:
                    break
                if not active_segments_only or (hasattr(s, 'activity') and s.activity):
                    if self.n_features == 1:
                        minibatch[count, 0, :] = s.features[self.feature_name].T
                    else:
                        minibatch[count, 0, :, :] = s.features[self.feature_name].T
                    if with_filenames:
                        filenames[count] = sc.audio_path
                    if with_targets:
                        targets[count] = s.label

                    count += 1
                    if count == self.batch_size:
                        count = 0
                        data = [minibatch]

                        # create new arrays (alternatively, arrays could be copied when yielded)
                        if self.n_features == 1:
                            minibatch = np.empty((self.batch_size, 1, self.n_time_bins),
                                                 dtype=np.float32)
                        else:
                            minibatch = np.empty((self.batch_size, 1, self.n_features, self.n_time_bins),
                                                 dtype=np.float32)

                        if with_targets:
                            data.append(targets)
                            targets = np.empty((self.batch_size), dtype=np.int16)
                        if with_filenames:
                            data.append(filenames)
                            filenames = np.empty((self.batch_size), dtype="|U200")
                        yield tuple(data)
