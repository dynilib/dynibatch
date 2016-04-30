import argparse
import logging
from libdyni.utils.segment_container import create_segment_containers_from_seg_files


logger = logging.getLogger(__name__)


def run(input_root, output_root):

    sc_gen = create_segment_containers_from_seg_files(input_root)

    for sc in sc_gen:
        sc.save(output_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create segment container from seg files.")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("input_root", help="Seg file root path.")
    parser.add_argument("output_root")
    args = parser.parse_args()

    logger.setLevel(args.loglevel)

    run(args.input_root, args.output_root)
