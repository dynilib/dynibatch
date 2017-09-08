Dynibatch's config file documentation
=====================================

Here we describe how to create your own configuration file.
Like in the tutorial we will use the JSON format.
Each part of this file is explained in this page.


**data_path_config** content
----------------------------

Each configuration file may have this block. The possible keys are as follow:

* **"audio_root"** (required): the root folder path to the audios

* **"file2label_filename"** or **"seg2label_root"** (optional):

  * **"file2label_filename"**: path to a file containing all labels

  * **"seg2label_root"**: path to a root folder containing all the label files

* **"datasplit_path"** (required): if not "", file path to a given datasplit (can be created with dynibatch.utils.datasplit_utils)

* **"features_root"** (optional): if not "", the computed features will be cached

Example:

.. code-block:: json

  {
      "data_path_config": {
          "audio_root": "data/audio",
          "file2label_filename": "data/labels.csv",
          "datasplit_path": "data/datasplit_1504196263.jl",
          "features_root": "data/features/"
      },

      "others"
  }



**data_path_config** content
----------------------------

Each configuration file may have this block. The possible keys are as follow:

* **"batch_size"** (required): the number of examples per batch
* **"randomize_batch"** (required):
      0 means the files are taken without shuffle and sequentially

      1 means shuffle of all files at each epoch and taken sequentially

Example:

.. code-block:: json

  {
      "others",

      "minibatch_config": {
          "batch_size":10,
          "randomize_batch":1
      },

      "others"
  }

**segment_config** content
--------------------------

TODO

Example:

.. code-block:: json

  {
      "others",

      "segment_config" : {
          "seg_duration": 0.2,
          "seg_overlap": 0.5
      },

      "others"
  }

**audio_frame_config** content
------------------------------

TODO

Example:

.. code-block:: json

  {
      "others",

      "audio_frame_config" : {
          "sample_rate": 44100,
          "win_size":512,
          "hop_size":256
      },

      "others"
  }

**activity_detection** content
------------------------------

This optional block allow to configure an activity detector that can be use when generating data (conditionally to the detections).

Example:

.. code-block:: json

  {
      "others",

      "activity_detection" : {
          "name": "simple",
          "config":{
              "energy_threshold":0.2,
              "spectral_flatness_threshold":0.3
          }
      },

      "others"
  }

**features** content
--------------------

It is a list of blocks describing each feature to compute for the user.
TODO list them; in practice all the features in dynibatch.features.extractors

Example:

.. code-block:: json

  {
      "others",

      "features": [
          {
              "name": "mel_spectrum",
              "config": {
                  "n_mels": 56,
                  "min_freq": 50,
                  "max_freq": 22050
              }
          }
      ]
  }
