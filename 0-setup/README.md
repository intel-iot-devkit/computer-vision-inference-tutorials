# Intel® Computer Vision SDK Setup for using the Inference Engine

## What you’ll learn
  * How to install the OpenCL Runtime Package
  * How to install the Intel® CV SDK 

## Gather your materials
  * 5th or greater Generation Intel® Core™ processor. You can find the product name in Linux\* by running the ‘lscpu’ command. The ‘Model name:’ contains the information about the processor.

**Note**: The generation number is embedded into the product name, right after the ‘i3’, ‘i5’, or ‘i7’.  For example, the Intel® Core™ i5-5200U processor and the Intel® Core™ i5-5675R processor are both 5th generation, and the Intel® Core™ i5-6600K processor and the Intel® Core™ i5 6360U processor are both 6th generation.

  * Ubuntu\* 16.04.3 LTS
  * In order to run inference on the integrated GPU:  
	* A processor with Intel® Iris® Pro graphics or HD Graphics 
	* No discrete graphics card installed (required by OpenCL™).  If you have one, make sure to disable it in BIOS before going through this installation process.
	* No drivers for other GPUs installed, or libraries built with support for other GPUs   
	
## Install OpenCL Runtime Package
In order to run inference on the GPU, you need to first install the OpenCL runtime package. These commands install the OpenCL runtime package, as well as some package dependencies required by the CV SDK. 

**Note:** These steps are for Ubuntu 16.04.3 or later.  If you have a version older than 16.04.3, then you need to still install the Package dependencies below, then skip down to the Install Intel CV SDK section for instructions on installing the OpenCL driver.

Package dependencies:
```
sudo apt-get update
sudo apt-get install build-essential ffmpeg cmake checkinstall pkg-config yasm libjpeg-dev curl imagemagick gedit mplayer unzip libpng12-dev libcairo2-dev libpango1.0-dev libgtk2.0-dev libgstreamer0.10-dev libswscale.dev libavcodec-dev libavformat-dev
```

OpenCL runtime package:
```
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/11396/SRB5.0_linux64.zip
unzip SRB5.0_linux64.zip -d SRB5.0_linux64
cd SRB5.0_linux64
```
``` 
sudo apt-get install xz-utils
mkdir intel-opencl
tar -C intel-opencl -Jxf intel-opencl-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf intel-opencl-devel-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf intel-opencl-cpu-r5.0-63503.x86_64.tar.xz
sudo cp -R intel-opencl/* /
sudo ldconfig
```

## Install Intel® CV SDK
1. Go to https://software.seek.intel.com/computer-vision-software
2. Register, then wait for a confirmation email.  It can take *__several__* hours to get the email. So go take a break and come back once you're received the email. 

If you've already registered for the CV SDK before, you should get access to download almost immediately.

![](images/email-confirmation.jpg)
3. From the link in the email, download the __Ubuntu* package__
![](images/download-page-1.jpg)
4. Unzip the contents (to a folder in your directory of choice)
```
tar zxvf intel_cv_sdk_ubuntu_r3_2017.1.163.tgz
```

**Note:** If you are running a version older than Ubuntu 16.04.3, then to install the OpenCL driver you need to run the ```install_OCL_driver.sh``` script in the downloaded folder **before** running the CV SDK installation.  This script can take over half an hour to complete.  It will re-build the kernel with the updated driver.  Your computer will restart through the process. Make sure to backup your data before running this script.  If you prefer not to continue with this kernel re-build, then we recommend you install Ubuntu 16.04.3 or later which only needs a few files installed (instructions above) and does not need a kernel re-build.

5. In the cv sdk folder: 
```
cd intel_cv_sdk_ubuntu_r3_2017.1.163/
```
Enter super user mode  
```
sudo su
```
Then run the installation wizard  
```
./install_GUI.sh
```
and follow the instructions.

## Try a Sample Application Using the Inference Engine
One of the main advantages of the Intel® CV SDK is the Inference Engine, which also allows you to take advantage of the Intel® integrated GPU.  

Run the [Object Detection using Inference and SSD tutorial](../1-object-detection-ssd).


