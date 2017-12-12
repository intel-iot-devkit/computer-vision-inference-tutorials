# Intel® Computer Vision SDK Setup for using the Inference Engine

## What you’ll learn
  * How to check your system requirements for a Intel® Computer Vision SDK installation.
  * How to install the Intel® CV SDK with optional GPU capabilities 
  * How to verify the Intel® CV SDK installation 

## Gather your materials
  * 5th or 6th Generation Intel® Core™ processor. You can find the product name in Linux\* by running the ‘lscpu’ command. The ‘Model name:’ contains the information about the processor.

**Note**: The generation number is embedded into the product name, right after the ‘i3’, ‘i5’, or ‘i7’.  For example, the Intel® Core™ i5-5200U processor and the Intel® Core™ i5-5675R processor are both 5th generation, and the Intel® Core™ i5-6600K processor and the Intel® Core™ i5 6360U processor are both 6th generation.

  * Ubuntu\* 16.04.2 LTS
  * Requirements to run inference on the GPU (optional):  
	* A processor with Intel® Iris® Pro graphics and HD Graphics **HOW TO CHECK?** 
	* No discrete graphics card installed (required by OpenCL™)
	* No drivers for other GPUs installed, or libraries built with support for other GPUs **HOW TO CHECK?** 

**Note**: It isn't required, but it's easier to install it now if you think you may want it.  Having it installed can be useful to compare the CPU and GPU performance of inference.

## Setup
### Install OpenCL Runtime Package
```
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/11396/SRB5.0_linux64.zip

unzip SRB5.0_linux64.zip -d SRB5.0_linux64

cd SRB5.0_linux64

sudo apt-get update
sudo apt-get install xz-utils
mkdir intel-opencl
tar -C intel-opencl -Jxf intel-opencl-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf intel-opencl-devel-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf intel-opencl-cpu-r5.0-63503.x86_64.tar.xz
sudo cp -R intel-opencl/* /
sudo ldconfig
```

### Install required dependencies:
```
apt-get update
apt-get -y install build-essential cmake libopencv-dev checkinstall pkg-config yasm libjpeg-dev curl imagemagick gedit mplayer libgstreamer0.10-dev  
```

### Install Intel® CV SDK
1. Go to https://software.seek.intel.com/computer-vision-software
2. Register, then download the __Ubuntu* package__
![](images/download-page-1.jpg)
3. Unzip the contents (to a folder in your directory of choice)
4. From the folder run through the installation wizard
```
./install_GUI.sh
```
![](images/installation-wizard.png)

### Verify Intel® CV SDK Installation

### Try a Sample Application Using the Inference Engine
One of the main advantages of the Intel® CV SDK is the Inference Engine, which also allows you to take advantage of the Intel® integrated GPU.  

Run the [Object Detection using Inference and SSD tutorial](../1-object-detection-ssd).


