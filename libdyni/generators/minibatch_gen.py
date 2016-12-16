import logging
import json
import numpy as np

# generators
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
# features
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor

from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
# activity detection
from libdyni.features.activity_detection.simple import Simple
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
            Create a minibatch generator from JSON file
            Args:
                config_path: path to the JSON config file
            Returns a minibatch generator
        """

        frame_feature_extractors = set()

        # parse json file
        with open(config_path) as config_file:
            config = json.loads(config_file.read())

        # audio and short-term frames config
        sample_rate = config["sample_rate"]
        win_size = config["win_size"]
        hop_size = config["hop_size"]

        # segment config
        seg_duration = config["seg_duration"]

        # minibatch config
        batch_size = config["batch_size"]
        num_frames_per_seg = int(seg_duration * sample_rate / hop_size)

        # create a parser to get the labels from the label file
        label_parser = label_parsers.CSVLabelParser(config["label_file_path"])

        # get activity detection
        if "activity_detection" in config:
            act_det_config = config["activity_detection"]

            if act_det_config["name"] == "simple":

                # frame features needed for the activity detection
                en_ext = EnergyExtractor()
                sf_ext = SpectralFlatnessExtractor()
                frame_feature_extractors |= set([en_ext, sf_ext])

                act_det = Simple(
                    energy_threshold=act_det_config["energy_threshold"],
                    spectral_flatness_threshold=act_det_config["spectral_flatness_threshold"])

            else:
                raise exceptions.LibdyniError("Activity detector {} not supported".format(
                    act_det_config['name']))
        else:
            act_det = None

        # get feature that will feed the minibatch
        feat_config = config["feature"]

        if feat_config['name'] == 'mel':
            # mel spectra config
            n_mels = feat_config["n_mels"]
            min_freq = feat_config["min_freq"]
            max_freq = feat_config["max_freq"]
            feature = MelSpectrumExtractor(
                sample_rate=sample_rate,
                fft_size=win_size,
                n_mels=n_mels,
                min_freq=min_freq,
                max_freq=max_freq)
        else:
            raise exceptions.LibdyniError("Feature {} not supported".format(
                feat_config['name']))

        frame_feature_extractors.add(feature)

        # create a frame feature processor, in charge of computing all short-term features
        ff_pro = FrameFeatureProcessor(
            AudioFrameGen(win_size=win_size, hop_size=hop_size),
            frame_feature_extractors
        )

        # create needed segment-based feature extractors
        ffc_ext = FrameFeatureChunkExtractor(feature.name)

        # create a segment feature processor, in charge of computing all segment-based features
        sf_pro = SegmentFeatureProcessor(
            [act_det, ffc_ext],
            ff_pro=ff_pro,
            audio_root=config["audio_root"])

        # create and start the segment container generator that will use all the objects
        # above to generate for every audio files a segment container containing the list
        # of segments with the labels, the feature and an "activity detected" boolean
        # attribute
        sc_gen = SegmentContainerGenerator(
            config["audio_root"],
            sf_pro,
            label_parser=label_parser,
            seg_duration=config["seg_duration"],
            seg_overlap=config["seg_overlap"],
            random_order=config["random_batch"])

        return  MiniBatchGen(sc_gen,
                             feature.name,
                             batch_size,
                             feature.size,
                             num_frames_per_seg)


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
