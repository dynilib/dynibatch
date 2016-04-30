import os
import argparse
import logging
from libdyni.generators import audio_frame_gen, segment_feature_gen, frame_feature_gen
from libdyni.features import energy_extractor, spectral_flatness, activity_detection
from libdyni.utils.segment_container import SegmentContainer, load_segment_containers_from_dir, SC_EXTENSION
from libdyni.utils.segment import set_segment_labels, labels
from libdyni.utils.segment_container import create_segment_containers_from_audio_files


logger = logging.getLogger(__name__)


def run(audio_root, ref_seg_root, seg_duration, ref_seg_overlap,
        energy_threshold, spectral_flatness_threshold, feature_container_root=None):

    # create needed generators and feature extractors
    af_gen = audio_frame_gen.AudioFrameGen(win_size=512, hop_size=256)
    en_ext = energy_extractor.EnergyExtractor()
    sf_ext = spectral_flatness.SpectralFlatnessExtractor()
    ff_gen = frame_feature_gen.FrameFeatureGen(af_gen, [en_ext, sf_ext], feature_container_root)

    a_det = activity_detection.ActivityDetection(
            energy_threshold=energy_threshold,
            spectral_flatness_threshold=spectral_flatness_threshold)
    sf_gen = segment_feature_gen.SegmentFeatureGen([a_det],
            ff_gen=ff_gen,
            audio_root=audio_root)
    

    # stats
    ground_truth_activity = 0
    ground_truth_no_activity = 0
    detection_activity = 0
    detection_no_activity = 0

    # variables required to compute precision and recall
    true_pos = 0
    false_pos = 0
    false_neg = 0

    # process
    auto_sc_gen = create_segment_containers_from_audio_files(audio_root, seg_duration=seg_duration)
    for auto_sc in auto_sc_gen:

        ref_sc_path = os.path.join(
                ref_seg_root,
                os.path.splitext(os.path.basename(auto_sc.audio_path))[0] +
                SC_EXTENSION)

        try:
            ref_sc = SegmentContainer.load(ref_sc_path)
        except FileNotFoundError:
            logger.warning("Segment container file {} not found".format(ref_sc_path))
            continue

        # set label according to ground truth
        set_segment_labels(ref_sc.segments, auto_sc.segments, overlap_ratio=ref_seg_overlap)

        # detect activity
        sf_gen.execute(auto_sc)


        for s in auto_sc.segments:
            if s.label == labels.unknown:
                ground_truth_no_activity += 1
            else:
                ground_truth_activity += 1
            if s.activity:
                detection_activity += 1
            else:
                detection_no_activity += 1

            if s.activity and not s.label == labels.unknown:
                true_pos += 1
            elif s.activity and s.label == labels.unknown:
                false_pos += 1
            elif not s.activity and not s.label == labels.unknown:
                false_neg += 1

    precision = float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)

    return (ground_truth_activity, ground_truth_no_activity,
            detection_activity, detection_no_activity,
            precision, recall)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the activity detection on Sermicro DB.")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("audio_root", help="Audio files root path.")
    parser.add_argument("ref_seg_root", help="Reference segments root path.")
    parser.add_argument("seg_duration", type=float, help="Duration of the automatically splitted fixed-length segments.")
    parser.add_argument("ref_seg_overlap", type=float, help="Overlap ratio (in ]0,1[) to map labels of reference segments to automatic segments.")
    parser.add_argument("energy_threshold", type=float, help="Energy ratio threshold for activity detection.")
    parser.add_argument("spectral_flatness_threshold", type=float, help="Spectral flatness threshold for activity detection.")
    parser.add_argument("--fc_root", help="Feature containers path.", default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    results = run(args.audio_root, args.ref_seg_root, args.seg_duration, args.ref_seg_overlap,
            args.energy_threshold, args.spectral_flatness_threshold, args.fc_root)

    print("Ground truth: {0} with activity, {1} with no activity".format(
        results[0], results[1]))
    print("Detection: {0} with activity, {1} with no activity".format(
        results[2], results[3]))
    print("Precision: {}".format(results[4]))
    print("Recall: {}".format(results[5]))


