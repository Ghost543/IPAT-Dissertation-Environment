import sys


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_core_imports():
    import numpy
    import pandas
    import scipy
    import sklearn
    import h5py
    import yaml

