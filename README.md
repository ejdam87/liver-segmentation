# Liver segmentation
This repository contains a pipeline for segmenting liver from grayscale CT scans.

## Dependencies
- [PyTorch](https://pytorch.org/) library for neural network building blocks
- [NumPy](https://numpy.org/doc/) library for numerical computation
- [Pandas](https://pandas.pydata.org/docs/) library for data manipulation
- [Scikit-learn](https://scikit-learn.org/) library for train-test split
- [TQDM](https://tqdm.github.io/) library for progress visualization
- [Pillow](https://pillow.readthedocs.io/en/stable/) library for image processing

We use [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit for GPU acceleration, thus, make sure to install the version of PyTorch supporting it.

## Execution
- To run model training with train configuration in `configs` directory, execute `python main.py train`
- To run model testing with test configuration in `configs` directory, execute `python main.py test`
- To run model prediction with prediction configuration in `configs` directory, execute `python main.py pred`

## Models
As our main segmentation model, we use [U-Net](https://arxiv.org/abs/1505.04597) architecture.

As our baseline model, we use a CNN architecture, with a succession of 3x3 convolutional layers (associated with ReLU) and pooling layers. The last layers do upsample, convolution 1x1 and activation function.

## Data
The data can be accessed via this [URL](https://drive.google.com/file/d/1nQ6Sh_Y8rbP_m6j2xUb7zvSV0-XY2d9c/view). They are axial human CT scans. The dataset contains both inputs and also corresponding anotated expected outputs (labels).

## Results
Results are presented in `report.pdf` in great detail.
