.. Dynibatch documentation master file, created by
   sphinx-quickstart on Fri Sep  8 17:14:58 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dynibatch's documentation
====================================

Dynibatch is a Python library dedicated to providing mini-batches of audio data to machine learning algorithms.

It has been designed to deal with the following issues:

* **Reproducibility**: given a dataset (data + train/valid/test split + labels) and some parameters (e.g. segment size and overlap, audio feature to be used...), an experiment should be easily reproducible. Dynibatch allows to keep all this information in dedicated objects and a configuration file.

* **Big data**: because some datasets are huge, Dynibatch keeps a low memory footprint by generating mini-batches on the fly. To avoid recomputing at every epoch the data needed to generate the mini-batches, it can be cached in the disk.

* **Label management**: labels are typically provided either *per file* (i.e. one label for the whole audio file) or *per chunk* (i.e. one label for an audio chunk, delimited by a start time and an end time). In either case, Dynibatch automatically maps the labels provided with the dataset to *one label per segment*, where a segment is one fixed-size observation (see [the tutorial](examples/tutorial.ipynb) for more details). Labels are not mandatory, so that unsupervised algorithms can be run.

* **Usability**: with a given config file, generating mini-batches is as easy as

  .. code-block:: python

    mb_gen = MiniBatchGen.from_config(config)
    mb_gen_e = mb_gen.execute(with_targets=True)

    # get the first mini-batch
    data, targets = next(mb_gen_e)


========================
Starting using dynibatch
========================

.. toctree::
   :maxdepth: 2

   config_file.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
