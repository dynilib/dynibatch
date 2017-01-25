def get_n_overlapping_chunks(data_size, chunk_size, chunk_overlap):
    """Compute the number of overlapping chunks

    Args:
        data_size (int)
        chunk_size (int)
        chunk_overlap (int)
    Returns:
        The number of chunks.
    """

    hop_size = chunk_size * (1 - chunk_overlap)
    return int((data_size - chunk_size) / hop_size + 1)
