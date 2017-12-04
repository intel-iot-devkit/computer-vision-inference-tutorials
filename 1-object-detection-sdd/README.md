# Object Detection using Inference and SDD

This tutorial will walk you through the basics of using the Deep Learning Inference Engine included in the Intel Computer Vision SDK (CV SDK).  Inference includes using a trained neural network and feeding it an image to get the results.  Inference is typically done using a neural network architecture, such as AlexNet, GoogleNet, or Single Shot MultiBox Detector (SSD), which can be ran on various frameworks, like Caffe, Tensorflow, Torch, and more.  This example uses the Single Shot MultiBox Detector (SSD) with Caffe.

### So what's different about running a neural network on the Deep Learning Inference Engine versus the out of the box framework?  
* The Deep Learning Inference Engine optimizes the model to run *__significantly faster__* on Intel Architecture.
* It also allows inference to be ran on other harware, not just the CPU, such as the built-in Intel GPU or FPGA accelerator card.

### How does the Deep Learning Inference Engine work?
The Inference Engine takes a neural network model and optimizes it to take advantage of advanced Intel instruction sets in the CPU, and also makes it compatible with the other hardware accelerators (GPU and FPGA).  To do this, the model files, e.g., .caffemodel, .prototxt, are given to the Model Optimizer.  The Model Optimizer then processes the files and outputs two new files: a .bin and .xml.  These two files are used instead of the original model files when you run your application. In this example, the .bin and .xml files are already provided for you.

![](/images/inference_engine.jpg)

**WHAT DOES IR STAND FOR?**

## What you’ll learn
  * <>

## Gather your materials
  *	<>

## Setup
1. 

## Get the Code
<>

## How it works
<>


