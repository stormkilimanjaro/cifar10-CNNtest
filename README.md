# cifar10-CNNtest

Using the Tensorflow deep-learning framework and the CIFAR-10 dataset to build a classification model based on a 
convolutional neural network.

This code is based upon the following references:

* [A Step by Step Convolutional Neural Network using Tensorflow](http://tanzimsaqib.com/redir.html?https://raw.githubusercontent.com/tsaqib/ml-playground/master/cnn-tensorflow/cnn-tensorflow.html?v002) 
* [CIFAR-10 Image Classification in Tensorflow](https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c) 


## Basic information

### How to run it
1. Ensure that you're running Python 3.6 or Create a virtual environment with Python 3.6
2. Download the code from the repo into a directory
3. Download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
4. Unzip the dataset and store it in a subdirectory named `cifar-10`

### Requirements
* Numpy
* Pickle
* Tensorflow
* tqdm
* sklearn
* matplotlib


Install the packages above using ```python3.6 -m pip install [package name]```

### Running the training model
```
$ python code.py
```
### Testing the model
```
$ python test.py
```



