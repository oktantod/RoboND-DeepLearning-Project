[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project - Follow Me ##

In this projectwe will train a deep neural network, especially Fully Convolutional Neural Network (FCN) to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/followme.jpg
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

We used above data training and validation for train weight for FCN.

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

# Software used for Project Training:

* Windows 8.1 64bit
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

# Hardware used for Project Training:

* Notebook ASUS N56V
- Intel(R) Core(TM) i7 - 3630QM
- RAM 8 GB
* NVIDIA GeForce 650M (2 GB with 384 core)

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Write Up Report Follow Me Project
## Write Up by Dedi

# Data Collection

In this learning project, I didn't record train, validation and sample_evaluation_data data from quadcopter simulator. I used train and validation data from link above to get weight from the network model that I have design. 

<table><tbody>
    <tr><th align="center" colspan="3"> Data Set 1</td></tr>
    <tr><th align="center">Folder</th><th align="center">Content</th></tr>
    <tr><td align="left">/data/train</td><td align="left">4,131 images + 4,131 masks</td></tr>
    <tr><td align="left">/data/validation</td><td align="left">1,184 images + 1,184 masks</td></tr>    
    <tr><td align="left">/data/sample_evalution_data/following_images</td>
       <td align="left">542 images + 542 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_non_targ</td>
       <td align="left"> 270 images + 270 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_with_targ</td>
       <td align="left"> 322 images + 322 masks</td></tr>
</tbody></table>

# FCN Layers

In this project, there are seven layers to build a fully convolutional networks (FCN). Three layers for encoder, one layers as one by one convolutional matrix and another three layers as decoder block. See image below:

<p align="center"> <img src="./docs/misc/FCN Design.jpg"> </p>

* The first layer is encoder block layer which have input from image. This layer have filter width 32 and 2 strides.
* The second layer is encoder block layer which have input from first layer. This layer have filter width 64 and 2 strides.
* The third layer is encoder block layer which have input from second layer. This layer have filter with width 128 and build from 2 strides.
* The fourth layer is 1x1 convolution layer using convolution 2D batch normalize. This layer build with filter width 256, and kernel 1x1 with 1 strides.
* The fifth layer is decoder block layer which have input from 1x1 convolution layer and skip connection from second layer.
* The sixth layer is decoder block layer which have input from fifth layer and skip connection from the first layer. The dimension of this layer is same with first layer.
* The last layer is decoder block layer which have input from sixth layer and skip connection from input image. The last layer is an output layer of FCN. This layer have the dimension as same as input image.

Explanations of how to build the code for FCN Design above would be explain in Build the Model section below

# Build The Model

### Separable convolution layer:
The Encoder for FCN require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for your use. Each includes batch normalization with the ReLU activation function applied to the layers.
```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

### Bilinear Upsampling
The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended, but you can try out different factors as well. Upsampling is used in the decoder block of the FCN.
```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

## TODO Code

### Encoder Block
Create an encoder block that includes a separable convolution layer using the separable_conv2d_batchnorm() function. The filters parameter defines the size or depth of the output layer.
```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

### Decoder Block
The decoder block is comprised of three parts:
*    A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
*    A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
*    Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample_small_ip_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsample_small_ip_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    
    return output_layer
```

### The FCN Model
Now that you have the encoder and decoder blocks ready, go ahead and build your FCN architecture!

There are three steps:

*    Add encoder blocks to build the encoder layers. This is similar to how you added regular convolutional layers in your CNN lab.
*    Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
*    Add decoder blocks for the decoder layers.

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    layer01 = encoder_block(inputs , filters=32 , strides=2)
    
    layer02 = encoder_block(layer01, filters=64 , strides=2)
    
    layer03 = encoder_block(layer02, filters=128, strides=2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    layer04 = conv2d_batchnorm(layer03, filters=256, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    layer05 = decoder_block(layer04, layer02, filters=128 )
    
    layer06 = decoder_block(layer05, layer01, filters=64  )
    
    layer07 = decoder_block(layer06, inputs , filters=32  )
    
    # The function returns the output layer of your model. "layer07" is the final layer obtained from the last decoder_block()
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer07)
    print("Outputs shape:",outputs.shape, "\tOutput Size in Pixel")
    
    return outputs
```

#Training
Training is my bigest problems. I have facing two problem in AWS account, the first one is AWS reject my request increasing EC2 instance p2.xlarge and the second ones is AWS facing problem when send my promotion code for initial balance by mail. Thanks for support dashboard in AWS, my complain had been approved at 18 June for initial credit. And at 20 June AWS approved for increasing limit in p2.xlarge when I reopen the case. Sadly when all my request had been approved, I moved to my village with lower internet connectivity speed therefore I used my laptop for trained the model. Spesification of my laptop you can see at above explanations.

To increase the speed of training in my laptop, I install the following software and library:
* Latest NVIDIA Driver 398.11
* CUDA v9.0
* CuDNN v7.1

## Hyperparameters

### Batch Size
Number of training samples/images that get propagated through the network in a single pass. In this training we used batch_size with value 32.
```python
```


# Test the model that have been created in the quadcopter simulator

The model weights selected is model_weights_new that have final score 40.74, to run this model weight

```bash
>python follower.py model_weights_new
```
[[Simulation Video]](https://youtu.be/cotA3RwEjA8)

