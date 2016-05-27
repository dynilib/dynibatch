import argparse
import logging
import joblib

from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.features.energy_extractor import EnergyExtractor
from libdyni.features.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.chirplets_chunk_extractor import ChirpletsChunkExtractor
from libdyni.features.activity_detection import ActivityDetection
from libdyni.utils.neuralnet import build_cnn_chirplets, train_cnn
from libdyni.parsers import label_parsers

logger = logging.getLogger(__name__)


def run(audio_root,
        feature_container_root,
        chirplets_root,
        label_parser,
        classes,
        datasplit=None,
        scaler=None,
        sample_rate=22050,
        win_size=256,
        hop_size=128,
        energy_threshold=0.2,
        spectral_flatness_threshold=0.3,
        seg_duration=0.5,
        seg_overlap=0.9,
        batch_size=10,
        num_features=1,
        num_time_bins=11025,
        learning_rate=0.1,
        reg_coef=0,
        num_epochs=1000):
    # create needed generators and feature extractors
    af_gen = AudioFrameGen(win_size=win_size, hop_size=hop_size)

    en_ext = EnergyExtractor()
    sf_ext = SpectralFlatnessExtractor()
    ff_pro = FrameFeatureProcessor(
        af_gen,
        [en_ext, sf_ext],
        feature_container_root)

    ch_ext = ChirpletsChunkExtractor(sample_rate, chirplets_root, scaler)
    act_det = ActivityDetection(
        energy_threshold=energy_threshold,
        spectral_flatness_threshold=spectral_flatness_threshold)
    sf_pro = SegmentFeatureProcessor(
        [act_det, ch_ext],
        ff_pro=ff_pro,
        audio_root=audio_root)

    # create segment container generator
    # TODO shuffle inputs (not trivial because data are generated
    #      from os.walk which always returns the data in the same order
    sc_gen_train = SegmentContainerGenerator(
        audio_root,
        label_parser,
        sf_pro,
        dataset=datasplit["sets"]["train"],
        seg_duration=seg_duration,
        seg_overlap=seg_overlap)

    sc_gen_valid = SegmentContainerGenerator(
        audio_root,
        label_parser,
        sf_pro,
        dataset=datasplit["sets"]["validation"],
        seg_duration=seg_duration,
        seg_overlap=seg_overlap)

    sc_gen_test = SegmentContainerGenerator(
        audio_root,
        label_parser,
        sf_pro,
        dataset=datasplit["sets"]["test"],
        seg_duration=seg_duration,
        seg_overlap=seg_overlap)

    logger.debug("Building network...")

    # build network
    input_var, target_var, network = build_cnn_chirplets(batch_size,
                                                         num_features,
                                                         num_time_bins,
                                                         len(classes))

    logger.debug("Training network...")

    # train network
    train_cnn(sc_gen_train,
              sc_gen_valid,
              sc_gen_test,
              classes,
              "chirplets",
              network,
              input_var,
              target_var,
              num_epochs,
              batch_size,
              num_features,
              num_time_bins,
              learning_rate,
              reg_coef)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN with raw audio.")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("audio_root", help="Audio files root path.")
    parser.add_argument("features_root", help="Feature containers root path.")
    parser.add_argument("chirplets_root", help="Chirplets root path.")
    parser.add_argument("datasplit_path", help="Data split path.")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    audio_root = args.audio_root
    features_root = args.features_root
    chirplets_root = args.chirplets_root
    datasplit_path = args.datasplit_path
    learning_rate = args.learning_rate

    datasplit = joblib.load(datasplit_path)

    # ugly hardcoded stuff to be sorted out
    parser = label_parsers.Bird10Parser("/home/ricard/data/Bird10/labels/labels.csv")
    # classes = sorted(["45", "09", "26"])
    classes = sorted(['45', '09', '26', '10', '31', '35', '05', '40', '36', '27'])
    scaler = joblib.load("/home/ricard/data/Bird10/chirplets/scaler/scaler.jl")

    run(audio_root,
        features_root,
        chirplets_root,
        parser,
        classes,
        datasplit,
        scaler=scaler,
        sample_rate=22050,
        win_size=256,
        hop_size=128,
        energy_threshold=0.2,
        spectral_flatness_threshold=0.3,
        seg_duration=0.5,
        seg_overlap=0.9,
        batch_size=50,
        num_features=80,
        num_time_bins=110,
        learning_rate=learning_rate,
        reg_coef=0.0001,
        num_epochs=1000)
