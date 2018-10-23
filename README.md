# GoogLeNet for Image Classification

- TensorFlow implementation of [Going Deeper with Convolutions](https://research.google.com/pubs/pub43022.html) (CVPR'15). 
<!-- - **The inception structure** -->
- This repository contains the examples of natural image classification using pre-trained model as well as training a Inception network from scratch on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (93.64% accuracy on testing set). The pre-trained model on CIFAR-10 can be download from [here](https://www.dropbox.com/sh/kab0bzpy0zymljx/AAD2YCVm0J1Qmlor8EoPzgQda?dl=0).
- Architecture of GoogLeNet from the paper:
![googlenet](fig/arch.png)

## Requirements
- Python 3.3+
- [TensorFlow 1.9+](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)

## Usage

- Train
python3 inception_cifar.py --train --lr 0.01 --bsize 50 --keep_prob 0.2 --maxepoch 100
- Evaluate
python3 inception_cifar.py --eval --lr 0.01 --bsize 50 --keep_prob 0.2 --maxepoch 100

