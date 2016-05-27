import logging
import os
from os.path import join, splitext, relpath
import argparse
from libdyni.utils.segment_container import ALLOWED_AUDIO_EXT
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.energy_extractor import EnergyExtractor
from libdyni.features.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.mel_spectrum_extractor import MelSpectrumExtractor
from libdyni.features.mfcc_extractor import MFCCExtractor


logger = logging.getLogger(__name__)


WIN_SIZE = 256
HOP_SIZE = 128


def run(audio_root,
        feature_container_root,
        sample_rate):

    # create needed generators and feature extractors
    af_gen = AudioFrameGen(win_size=WIN_SIZE, hop_size=HOP_SIZE)

    en_ext = EnergyExtractor()
    sf_ext = SpectralFlatnessExtractor()
    mel_ext = MelSpectrumExtractor(sample_rate=sample_rate, fft_size=WIN_SIZE,
            n_mels=64, min_freq=0, max_freq=sample_rate/2)
    mfcc_ext = MFCCExtractor(sample_rate=sample_rate, fft_size=WIN_SIZE,
            n_mels=64, n_mfcc=16, min_freq=0, max_freq=sample_rate/2)
    ff_pro = FrameFeatureProcessor(af_gen, [en_ext, sf_ext, mel_ext, mfcc_ext], feature_container_root)

    for root, dirnames, filenames in os.walk(audio_root):

        for filename in filenames:

            basename, extension = splitext(filename)
            if not extension in ALLOWED_AUDIO_EXT: continue # only get audio files

            # compute features et write feature container
            ff_pro.execute((audio_root, relpath(join(root, filename), audio_root)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute and write features")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("audio_root", help="Audio files root path.")
    parser.add_argument("feature_container_root", help="Feature containers root path.")
    parser.add_argument("sample_rate", type=int, help="Sample rate.")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    run(args.audio_root,
            args.feature_container_root,
            args.sample_rate)
