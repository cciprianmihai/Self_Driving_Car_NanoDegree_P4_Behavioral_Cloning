

|Lake Track|
|[![Lake Track](images/lake_track.png)](https://youtu.be/y6GMAXvS22A)

## Project Description

### Objective: to build a neural network to clone car driving behaviour.
The goals / steps of this project are the following:

  * use the simulator to collect data of good driving behavior;
  * build, a convolution neural network in Keras that predicts steering angles from images;
  * train and validate the model with a training and validation set;
  * test that the model successfully drives around track one without leaving the road.
  
### General solution:
  * supervised regression problem (between the car steering angles and the road images in front of the car);
  * training dataset contains images that were taken from three different camera angles (Center, Left, Right of the car);
  * the neural network is based on the [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to give important results for this type of problem;
  * for the image processing step, the current model is using Convolutional layers (automated feature engineering).  

### Files included

- `model.py` The script used to create and train the model;
- `drive.py` The script to drive the car. You can feel free to resubmit the original `drive.py` or make modifications and submit your modified version. `drive.py` is originally from [the Udacity Behavioral Cloning project GitHub](https://github.com/udacity/CarND-Behavioral-Cloning-P3) but it has been modified to control the throttle;
- `utils.py` The script to provide useful functionalities (i.e. image preprocessing and augumentation);
- `model.h5` The model weights;
- `environments.yml` conda environment (Use TensorFlow without GPU);
- `environments-gpu.yml` conda environment (Use TensorFlow with GPU).

## Quick Start

### Install required python libraries:
This lab requires:

CarND Term1 Starter Kit: https://github.com/udacity/CarND-Term1-Starter-Kit
Unity Car Simulator: https://github.com/udacity/self-driving-car-sim
The lab enviroment can be created with CarND Term1 Starter Kit. 

In order to run this project, you need an [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python
# Use TensorFlow without GPU
conda env create -f environment.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries (see the contents of the environment*.yml files) using pip.

### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```
The car will use the trained model in order to follow the correct way.  

### To train the model

You'll need the data folder which contains the training images. We used the default [training dataset from udacity](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). Also, you can use the simulator to generate your own data for the training process ([training mode 1](https://www.youtube.com/watch?v=rKw8md-zVno), [training mode 2](https://www.youtube.com/watch?v=kTJiHXJe_t4)).

For more information, go to folder 'Documentation' in this repository.

In order to traing the model run the following command:
```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best. For example, the first epoch will generate a file called `model-000.h5`.

## Model Architecture Design

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA for the end-to-end self driving test.  As such, it is well suited for the project.  

The architecture consists in a Deep Convolution Network which works well with supervised image classification / regression problems.  As the NVIDIA model is well documented, we were able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

We have added the following adjustments to the model. 

- we used `Lambda layer` to normalized input images to avoid saturation and make gradients work better;
- we have added an additional `dropout layer` to avoid overfitting after the `convolution layers`;
- we have also included `ELU` for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the architecture model is presented below:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle.  However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction. Overall, the model is very functional to clone the given steering behavior.  

The below is a model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1   |
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1         |
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 1, 18, 64) |0       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |dropout_1        |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |252219  |                 |


## Data Preprocessing

### Image Sizing

  * the images are cropped so that the model won’t be trained with the sky and the car front parts;
  * the images are resized to 66x200 (3 YUV channels) as per NVIDIA model;
  * the images are normalized (image data divided by 127.5 and subtracted 1.0).  As stated in the Model Architecture section, this is to avoid saturation and make gradients work better).


## Model Training

### Image Augumentation

For training, we have used the following augumentation technique along with Python generator to generate unlimited number of images:

  * randomly choose right, left or center images;
  * for left image, steering angle is adjusted by +0.2;
  * for right image, steering angle is adjusted by -0.2;
  * randomly flip image left/right;
  * randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift);
  * randomly translate image vertically;
  * randomly added shadows;
  * randomly altering image brightness (lighter or darker).

Using the left/right images is useful to train the recovery driving scenario. The horizontal translation is useful for difficult curve handling (i.e. the one after the bridge).


### Examples of Augmented Images

The following is the example transformations:

**Center Image**

![Center Image](images/center.png)

**Left Image**

![Left Image](images/left.png)

**Right Image**

![Right Image](images/right.png)

**Flipped Image**

![Flipped Image](images/flip.png)

**Translated Image**

![Translated Image](images/trans.png)


## Training, Validation and Test

We have splitted the dataset into `train` and `validation` set in order to measure the performance at every epoch. Testing was done using the simulator implemented in `Unity`.

As for training process:
  * We used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image;
  * We used Adam optimizer for optimization with learning rate of `1.0e-4` which is smaller than the default of `1.0e-3`. The default value was too big and made the validation loss stop improving too soon;
  * We used `ModelCheckpoint` from `Keras` to save the model only if the validation loss is improved which is checked for every epoch.

### The Lake Side Track

As there can be unlimited number of images augmented, we set the samples per epoch to `20.000`. We tried from 1 to 200 epochs, but we found `5-10 epochs` is good enough to produce a well trained model for the lake side track. The batch size of `40` was chosen as that is the maximum size which does not cause out of memory error on out machine with `NVIDIA GeForce 1050TI` 4096 MB.

## Outcome

The model can drive the course without bumping into the side ways.

- [The Lake Track - YouTube Link](https://youtu.be/y6GMAXvS22A)

## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
