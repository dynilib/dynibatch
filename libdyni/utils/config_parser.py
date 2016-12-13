"""
    module to create a mionibatch generator from a config file
"""
import ast
# generators
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.generators.minibatch_gen import MiniBatchGen
# features
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor

from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
# activity detection
from libdyni.features.extractors.activity_detection import ActivityDetection
# utils
from libdyni.utils import label_parsers
from libdyni.utils import exceptions


def parse_config_file(config_path):
    """
        Create a minibatch generator from config_path file
        Args:
            config_path: path for a JSON file
        Returns a minibatch generator parametrise by config_path
    """

    # TODO deal with feaatureroot argument
    with open(config_path) as config_description:
        config_dict = ast.literal_eval(config_description.read())

    # Prepare config

    # audio and short-term frames config
    sample_rate = config_dict["sample_rate"]
    win_size = config_dict["win_size"]
    hop_size = config_dict["hop_size"]

    # activity detection config
    act_det, feat_det = config_activity_detection(config_dict["activity_detection"])

    # Create needed short-term (aka frame-based) feature extractors
    feat_minibatch = []
    n_features = 0
    for config_feature_dict in config_dict["features"]:
        feat_ext = config_feature(config_feature_dict, sample_rate, win_size)
        n_features += feat_ext.size
        feat_minibatch.append(feat_ext)

    # mini-batches config
    batch_size = config_dict["batch_size"]
    n_time_bins = int(config_dict["seg_duration"] * sample_rate / hop_size)

    # create a parser to get the labels from the labels.csv file
    label_parser = label_parsers.CSVLabelParser(config_dict["label_file_path"])

    # create a frame feature processor, in charge of computing all short-term features
    ff_pro = FrameFeatureProcessor(
        AudioFrameGen(win_size=win_size, hop_size=hop_size),
        feat_det + feat_minibatch
    )

    # create needed segment-based feature extractors
    ffc_ext = FrameFeatureChunkExtractor(feat_minibatch[0].name) # TODO deal with multiple features
    # create a segment feature processor, in charge of computing all segment-based features
    # (here only chunks of mel spectra sequences)
    sf_pro = SegmentFeatureProcessor(
        [act_det, ffc_ext],
        ff_pro=ff_pro,
        audio_root=config_dict["audio_root"])

    # create and start the segment container generator that will use all the objects
    # above to generate for every audio files a segment container containing the list
    # of segments with the labels, the feature and an "activity detected" boolean
    # attribute
    sc_gen = SegmentContainerGenerator(
        config_dict["audio_root"],
        sf_pro,
        label_parser=label_parser,
        seg_duration=config_dict["seg_duration"],
        seg_overlap=config_dict["seg_overlap"])
    sc_gen.start()

    # generate mini-batches
    mb_gen = MiniBatchGen(feat_minibatch[0].name, # TODO deal with multiple features
                          batch_size,
                          n_features,
                          n_time_bins)

    return mb_gen, sc_gen

def config_activity_detection(config_dict):
    """
        Args:
            config_dict: dictionnary with all parameters for activity detection
        Returns an instanciated activity detector and a list of features extrators required
    """

    if config_dict["name"] == "default":
        en_ext = EnergyExtractor() # needed for the activity detection
        sf_ext = SpectralFlatnessExtractor() # needed for the activity detection

        act_det = ActivityDetection(
            energy_threshold=config_dict["energy_threshold"],
            spectral_flatness_threshold=config_dict["spectral_flatness_threshold"])

        return act_det, [en_ext, sf_ext]
    else:
        raise exceptions.LibdyniError("Activity detector {} not supported".format(
            config_dict['name']))

def config_feature(config_dict, sample_rate, win_size):
    """
        Args:
            config_dict: dictionnary with all parameters for feature
        Returns an instanciated feature extractor
    """

    if config_dict['name'] == 'mel':
        # mel spectra config
        n_mels = config_dict["n_mels"]
        min_freq = config_dict["min_freq"]
        max_freq = config_dict["max_freq"]
        feature_ext = MelSpectrumExtractor(
            sample_rate=sample_rate,
            fft_size=win_size,
            n_mels=n_mels,
            min_freq=min_freq,
            max_freq=max_freq)
    else:
        raise exceptions.LibdyniError("Feature {} not supported".format(
            config_dict['name']))

    return feature_ext
