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
import numpy as np
import joblib

# generators
from dynibatch.generators.audio_frame_gen import AudioFrameGen
from dynibatch.generators.segment_container_gen import SegmentContainerGenerator
# features
from dynibatch.features.frame_feature_processor import FrameFeatureProcessor
from dynibatch.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from dynibatch.features.extractors.generic_feature_chunk import GenericChunkExtractor
from dynibatch.features.extractors.audio_chunk import AudioChunkExtractor
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

    The shape of the data for each feature in the generated minibatches is:
    - (batch_size, 1, feature_size, n_time_bins) when the data is 2D (e.g. mel bands)
    - (batch_size, 1, n_time_bins) when the data is 1D (e.g. audio)
    This follow Tensorflow's recommendation to use the NCHW format.

    """

    def __init__(self,
                 segment_container_gen,
                 batch_size,
                 feature_shape_dict):
        """Initializes minibatch generator.

        Args:
            segment_container_gen (SegmentContainerGenerator)
            batch_size (int): minibatch size in number of segments
            feature_shape_dict (dict): dictionnary (keys: feature_name) of
                dictionnary (keys: feature_size, n_time_bins)
                for the feature to pull from segments
        """

        self._segment_container_gen = segment_container_gen
        self._batch_size = batch_size
        self._feature_shape_dict = feature_shape_dict

    @classmethod
    def from_config(cls, config):
        """Creates a dict of minibatch generators from a config dict

        One minibatch generator is created for every set defined in the
        datasplit, if specified in the config file. Otherwise, only one is
        created.

        Args:
            config (dict): configuration object
        Returns:
            A dict with one item per set defined in the datasplit, such as
            out_dict["<set_name>"] = <minibatch generator for this set>. If no
            datasplit is defined, the set is named "default".
        """

        # data path config
        dp_config = config["data_path_config"]

        # minibatch config
        mb_config = config["minibatch_config"]

        # audio frame config
        af_config = config["audio_frame_config"]

        # segment config
        seg_config = config["segment_config"]

        # Create a label parser.
        # Because FileLabelParser is set with a file path and SegmentLabelParser
        # with a root path, two different keys are used
        if "file2label_filename" in dp_config:
            label_parser = label_parsers.CSVFileLabelParser(dp_config["file2label_filename"],
                                                            label_file=dp_config.get("label_file"))
        elif "seg2label_root" in dp_config:
            label_parser = label_parsers.CSVSegmentLabelParser(dp_config["seg2label_root"],
                                                               dp_config["label_file"])
        else:
            label_parser = None

        # get activity detection
        if "activity_detection" in config:
            act_det_config = config["activity_detection"]
            act_det = activity_detection.factory(
                act_det_config["name"],
                audio_frame_config=af_config,
                feature_config=act_det_config.get("config"))
        else:
            act_det = None

        # instanciate all frame feature extractors needed by the activity detection
        act_det_frame_feature_extractors = []
        if act_det:
            for ff_cfg in act_det.frame_feature_config:
                if extractors.is_feature_implemented(ff_cfg["name"]):
                    act_det_frame_feature_extractors.append(
                        extractors.factory(
                            ff_cfg["name"],
                            audio_frame_config=af_config,
                            feature_config=ff_cfg.get("config"))
                    )

        # instanciate all frame feature extractors that will feed the minibatch
        mb_frame_feature_extractors = []
        for ff_cfg in config["features"]:
            if (ff_cfg["name"] != "audio_chunk" and
                    extractors.is_feature_implemented(ff_cfg["name"])):
                mb_frame_feature_extractors.append(
                    extractors.factory(
                        ff_cfg["name"],
                        audio_frame_config=af_config,
                        feature_config=ff_cfg.get("config"))
                )


        # create a frame feature processor, in charge of computing all short-term features
        ff_pro = FrameFeatureProcessor(
            AudioFrameGen(
                sample_rate=af_config["sample_rate"],
                win_size=af_config["win_size"],
                hop_size=af_config["hop_size"]),
            act_det_frame_feature_extractors + mb_frame_feature_extractors,
            feature_container_root=dp_config.get('features_root')
        )

        # create needed segment-based feature extractors,
        # assuming that the extractor is a frame-based feature chunk extractor if it is
        # not implemented, and a generic chunk extractor otherwise
        sfe_list = []
        for feature_config in config['features']:
            if feature_config['name'] == "audio_chunk":
                sfe_list.append(
                    AudioChunkExtractor(dp_config['audio_root'], af_config['sample_rate'])
                )
            elif extractors.is_feature_implemented(feature_config['name']):
                if "scaler" in feature_config:
                    scaler = joblib.load(feature_config['scaler'])
                else:
                    scaler = None
                sfe_list.append(FrameFeatureChunkExtractor(feature_config['name'], scaler=scaler))
            else:
                sfe_list.append(GenericChunkExtractor(
                    feature_config['name'],
                    feature_config['sample_rate'],
                    feature_config['size']))

        # create a segment feature processor, in charge of computing all segment-based features
        sf_pro = SegmentFeatureProcessor(
            list(sfe_list + [act_det]) if act_det else sfe_list,
            ff_pro=ff_pro,
            audio_root=dp_config["audio_root"],
            feature_container_root=dp_config.get('features_root')
        )

        datasplit_path = dp_config.get("datasplit_path")
        sc_gen_dict = {}
        if not datasplit_path:
            # if no datasplit is present in the config file,
            # create one segment container generator
            sc_gen_dict["default"] = SegmentContainerGenerator(
                dp_config["audio_root"],
                sf_pro,
                label_parser=label_parser,
                seg_duration=seg_config["seg_duration"],
                seg_overlap=seg_config["seg_overlap"],
                shuffle_files=mb_config["shuffle_files"])
        else:
            # else create one per set in the datasplit
            datasplit = joblib.load(datasplit_path)
            for set_name, _ in datasplit["sets"].items():
                sc_gen_dict[set_name] = SegmentContainerGenerator(
                    dp_config["audio_root"],
                    sf_pro,
                    label_parser=label_parser,
                    dataset=datasplit["sets"][set_name],
                    seg_duration=seg_config["seg_duration"],
                    seg_overlap=seg_config["seg_overlap"],
                    shuffle_files=mb_config["shuffle_files"])

        # get all shape (feature_size, n_time_bins) for every feature
        # and build argument to be passed to the minibatch generators init
        # (i.e. a dict of dict {name: {feature size: value, num time bins: value}})
        feature_shape_dict = {}
        frame_feature_count = 0 # count frame features only, not audio_chunk
        for feature_config in config['features']:
            if feature_config['name'] == "audio_chunk":
                feature_size = 1
                n_time_bins = int(af_config["sample_rate"] * seg_config["seg_duration"])
            elif extractors.is_feature_implemented(feature_config['name']):
                feature_size = mb_frame_feature_extractors[frame_feature_count].size
                n_time_bins = int(seg_config["seg_duration"] *
                                  af_config["sample_rate"] / af_config["hop_size"])
                frame_feature_count += 1
            else:
                feature_size = feature_config["size"]
                n_time_bins = int(seg_config["seg_duration"] * feature_config["sample_rate"])
            feature_shape_dict[feature_config['name']] = {
                "feature_size": feature_size,
                "n_time_bins": n_time_bins
            }

        mb_gen_dict = {}
        for set_name, sc_gen in sc_gen_dict.items():
            mb_gen_dict[set_name] = MiniBatchGen(
                sc_gen,
                mb_config["batch_size"],
                feature_shape_dict)

        return mb_gen_dict

    @property
    def label_parser(self):
        """Return label_parser associated with the MiniBatchGen"""
        return self._segment_container_gen._label_parser


    def mb_feature_shape(self, name, data_format):
        """
        Get the shape of the data returned in the minibatch
        for the 'name' feature.

        Args:
            name (str): name of the feature
            data_format (str): either "NCHW" or "NHWC", where
                N = Num_samples
                H = Height
                W = Width
                C = Channels
        """

        if self._feature_shape_dict[name]["feature_size"] > 1:
            if data_format == "NCHW":
                return (self._batch_size,
                        1,
                        self._feature_shape_dict[name]["feature_size"],
                        self._feature_shape_dict[name]["n_time_bins"])
            else:
                return (self._batch_size,
                        self._feature_shape_dict[name]["feature_size"],
                        self._feature_shape_dict[name]["n_time_bins"],
                        1)
        else:
            if data_format == "NCHW":
                return (self._batch_size,
                        1,
                        self._feature_shape_dict[name]["n_time_bins"])
            else:
                return (self._batch_size,
                        self._feature_shape_dict[name]["n_time_bins"],
                        1)


    def execute(self,
                active_segments_only=False,
                known_labels_only=False,
                with_targets=False,
                with_filenames=False,
                data_format="NCHW"):
        """Executes the minibatch generator

        Args:
            active_segments_only (bool): returns only segments with "activity" attribute set to True
            known_labels_only (bool): returns only segments with label not set
                to segment.CommonLabels.unknown.value
            with_targets (bool): returns labels associated to the data
            with_filenames (bool): returns filenames where the data were taken
        Returns:
            tuple(features, targets (if with_targets), filenames (if with_filenames))
        """

        features = {key: np.empty(self.mb_feature_shape(key, data_format), dtype=np.float32)
                    for key in self._feature_shape_dict.keys()}

        if with_filenames:
            filenames = np.empty((self._batch_size), dtype="|U200")
        if with_targets:
            targets = np.empty((self._batch_size), dtype=np.int16)

        count = 0
        for sc in self._segment_container_gen.execute():
            logger.debug("iterate_minibatch: %s", sc.audio_path)
            for seg in sc.segments:
                missing_feature = False
                for feature_name in self._feature_shape_dict.keys():
                    if feature_name not in seg.features:
                        missing_feature = True
                if missing_feature:
                    break
                if ((not active_segments_only or (hasattr(seg, 'activity')) and seg.activity) and
                        (not known_labels_only or seg.label != CommonLabels.unknown.value)):

                    for feature_name in self._feature_shape_dict:
                        if self._feature_shape_dict[feature_name]["feature_size"] == 1:
                            if data_format == "NCHW":
                                features[feature_name][count, 0, :] = seg.features[feature_name].T
                            else:
                                features[feature_name][count, :, 0] = seg.features[feature_name].T
                        else:
                            if data_format == "NCHW":
                                features[feature_name][count, 0, :, :] = \
                                    seg.features[feature_name].T
                            else:
                                features[feature_name][count, :, :, 0] = \
                                    seg.features[feature_name].T
                    if with_filenames:
                        filenames[count] = sc.audio_path
                    if with_targets:
                        targets[count] = seg.label

                    count += 1
                    if count == self._batch_size:
                        count = 0
                        data = [features]

                        # create new array
                        features = {
                            key: np.empty(self.mb_feature_shape(key, data_format), dtype=np.float32)
                            for key in self._feature_shape_dict.keys()
                        }

                        if with_targets:
                            data.append(targets)
                            targets = np.empty((self._batch_size), dtype=np.int16)
                        if with_filenames:
                            data.append(filenames)
                            filenames = np.empty((self._batch_size), dtype="|U200")
                        yield tuple(data)
