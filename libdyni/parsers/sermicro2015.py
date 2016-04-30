import os


def get_label(audio_path):
    return os.basename(audio_path).split("_")[0]
