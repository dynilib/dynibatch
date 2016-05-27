import argparse
import logging
from libdyni.utils.segment_container import create_segment_containers_from_audio_files

logger = logging.getLogger(__name__)


def run(input_root, output_root, **kwargs):
    sc_gen = create_segment_containers_from_audio_files(input_root, **kwargs)

    for sc in sc_gen:
        sc.save(output_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create segment container from audio files.")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("input_root", help="Audio file root path.")
    parser.add_argument("output_root")
    parser.add_argument("--seg_duration", help="Duration of the segments (in s).", type=float)
    parser.add_argument("--seg_overlap", help="Overlap ratio of the segment (in ]0, 1])", type=float)
    args = parser.parse_args()

    logger.setLevel(args.loglevel)

    # build optional arguments
    opt_args = {}
    if args.seg_duration:
        opt_args["seg_duration"] = args.seg_duration
        if args.seg_overlap:
            opt_args["seg_overlap"] = args.seg_overlap

    run(args.input_root, args.output_root, **opt_args)
