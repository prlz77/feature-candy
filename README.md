# feature-candy
A feature visualization tool for CNN's. It is coded using PySide (Qt libraries) and currently only supports the Caffe backend. However, new backends can be easily added by using the abstract class called nnreader.

## Installation
Feature Candy works just executing the `feature_candy.py` script with python. Just make sure you setup the path to the backend in a `config.py` file and install the dependencies:

* PySide
* Scipy (with Numpy, Matplotlib, etc)
* Caffe (currently the only backend)

Enjoy the sweet sweet features!
