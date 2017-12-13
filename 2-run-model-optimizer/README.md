# Object Recognition using Inference and Single Shot MultiBox Detector (SSD)\*

This tutorial will walk you through the basics taking an existing model and converting it to IR (Intermediate Representation) files using the inference engine.

## What you’ll Learn
  * How to generate the .bin and .xml (IR files) needed for the Inference Engine from a Caffe model
  * Run the Inference Engine using the IR files in a C++ application

## Gather your materials
* Intel® Computer Vision SDK Beta r3 installed.  See [Setup insructions for how to install](../0-setup/) 
	
## Setup
If you have not already, install the Intel® Computer Vision SDK, see [Setup insructions for how to install](../0-setup/).

## Get the code
### Clone this repository
You should have already cloned this repository when you went through the setup, if not, you can get the code using
```
git clone https://github.com/intel-iot-devkit/computer-vision-inference-tutorials.git
```

## Install Caffe
From the `computer-vision-inference-tutorials/1-object-detection-ssd` directory type in a terminal:
```
sudo su

source /opt/intel/computer_vision_sdk_2017.1.163/bin/setupvars.sh

python installSSDCaffe.py

exit
```
This should take about 5 minutes.

## Generate the .bin and .xml (IR files) for the Inference Engine
The Caffe model files (SSD_GoogleNetV2_Deploy.prototxt and SSD_GoogleNetV2_Deploy.caffemodel) have already been provided for you in this folder.  You convert them to IR files by running the Model Optimizer using the runMO.py script.

sudo su

python runMO.py -w SSD_GoogleNetV2.caffemodel -d SSD_GoogleNetV2_Deploy.prototxt

exit
```


### Run the Inference Engine using the IR files in a C++ application

First build:
```
make
```
Then run:
```
./IEobjectdetection -i vtest.avi -fr 500 -m artifacts/VGG_VOC0712_SSD_300x300_deploy/VGG_VOC0712_SSD_300x300_deploy.xml -d CPU -t SSD -l pascal_voc_classes.txt
```

*Note* the vtest.avi video file is already included as part of this repository in the /videos folder 

You should see a video play with people walking across and red bouding boxes around them:

![](images/expected_results_1.jpg)

You should also see the output in the console showing the objects found and the confidence level.

![](images/expected_results_2.jpg)

Here is the mapping for the labels of the classification:

1 - plane  
2 - bicycle  
3 - bird  
4 - board  
5 - bottle  
6 - bus  
7 - car  
8 - cat  
9 - chair  
10 - cow  
11 - table  
12 - dog  
13 - horse  
14 - motorcycle  
15 - person  
16 - plant  
17 - sheep  
18 - sofa  
19 - train  
20 - monitor  

They can be found in ```pascal_voc_classes.txt``` in this folder.

In this sample video, there are approximately 790 frames, so by setting the number of frames to 790 you'll see the whole video.
```
-fr 790
```

If you change the threshold of the confidence level to 0.1, you'll see a lot more bounding boxes around the people, but also many false ones.
```
-thresh 0.1
```

![](images/expected_results_3.jpg)

## How it works

### How does the Inference Engine work?
The Inference Engine takes a representation of a neural network model and optimizes it to take advantage of advanced Intel instruction sets in the CPU, and also makes it compatible with the other hardware accelerators (GPU and FPGA). To do this, the model files (e.g., .caffemodel, .prototxt) are given to the Model Optimizer which then processes the files and outputs two new files: a .bin and .xml.  These two files are used instead of the original model files when you run your application. In this example, the .bin and .xml files are provided.

![](images/inference_engine.jpg)

In the above diagram, IR stands for Intermediate Representation, which is just a name for the .xml and .bin files that are inputs to the Inference Engine.

When you want to use another model in the Inference Engine, refer to this tutorial on how to get the necessary files using the Model Optimizer.
https://software.intel.com/en-us/inference-trained-models-with-intel-dl-deployment-toolkit-beta-2017r3

If you run into issues, don't hesitate to contact us on the forum https://software.intel.com/en-us/forums/computer-vision 

### High level steps in the code:

- Check arguments
- Video pre-processing
- Load model to the Inference Engine
- Run Inference
- Parse the results
- Render the frame

### Video pre-processing
The video pre-processing step takes the current frame, and resizes it using the `cv2.resize()` function, https://docs.opencv.org/3.0-beta/modules/imgproc/doc/geometric_transformations.html?highlight=resize#cv2.resize, 

It also converts it to a planar format (by default it is in packed format).  Packed format is arranged like RGBRGBRGB, planar is arranged like RRRGGGBBB.

This format is required by the Inference Engine because that is the format used by the Intel® Math Kernel Library (Intel® MKL).  If you want to know more see  
https://software.intel.com/en-us/ipp-dev-reference-pixel-and-planar-image-formats
and
https://software.intel.com/en-us/ipp-dev-guide-channel-and-planar-image-data-layouts

### Loading a model into the Inference Engine
There are three steps to loading a model into the Inference Engine: load the plugin, read the network and load the model into the plugin

1. Load the plugin
This is done by using the ```InferenceEnginePluginPtr _plugin()``` function.  

**__WHERE IS THIS PLUGIN LOCATED??__**

2. Read the network
The network object is initialized

```InferenceEngine::CNNNetReader network;```

and then a network is read using the ReadNetwork() function

```network.ReadNetwork(FLAGS_m)```

`FLAGS_m` means the `-m` flag, which in this case is ```SSD_GoogleNet_v2_fp32.xml```

The weights are then added to the network

```network.ReadWeights(binFileName.c_str());```

**Note**: The .xml file contains the network, and the .bin file contains the weights.

3. Load the model into the plugin

### Role of Model Optimizer
Model Optimizer is a command line tool that imports trained models...  optimizes a trained model

### Haar-based pedestrian detection

### Camera input

