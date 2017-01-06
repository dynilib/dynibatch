"""
    Module to facilitate the computation some information
"""

def get_n_overlapping_chunks(data_size, chunk_size, chunk_overlap):
    hop_size = chunk_size * (1 - chunk_overlap)
    return int((data_size - chunk_size) / hop_size + 1)
