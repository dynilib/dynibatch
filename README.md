# Libdyni

Libdyni is a library, written in Python, dedicated to audio data processing for machine learning tasks. One of its main objective is to ease the creation of minibatches (data + targets) to be fed to some machine learning algorithm.

## Install

These instructions have been tested on Ubuntu 14.04.

### Miniconda

Install Miniconda (a subset of Anaconda) with Python 3.5 following the instructions given in  
http://conda.pydata.org/miniconda.html.

### Get libdyni source

```
$ git clone https://<yourusername>@bitbucket.org/jul_dyni/libdyni.git
```

### Create conda environment

```
$ cd libdyni
$ conda env create -f libdyni.yml
```

### Activate conda environment

```
$ source activate libdyni
```

### Add libdyni to your PYTHONPATH

```
$ export PYTHONPATH=$PYTHONPATH:<path to libdyni>
```

### Test

Make sure all the tests pass.

```
$ cd <path to libdyni>
$ py.test tests
```

## Examples

TODO

## Dependencies

The list of dependencies is provided as information only, since they should all be installed during the creation of the libdyni conda environment. 
* [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org)
* [librosa](https://github.com/librosa/librosa)
* [PySoundFile](https://github.com/bastibe/PySoundFile)

## License

TODO
