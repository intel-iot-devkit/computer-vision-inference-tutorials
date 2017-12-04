#!/bin/bash

if [[ $EUID -ne 0 ]]; then
   echo "ERROR: you must run this script as a root." >&2
   echo "Please try again with "sudo -E $0", or as a root." >&2
   exit $EXIT_FAILURE
fi

#--------
# install prerequisites

echo "---------------------------"
echo "installing prerequisites..."
echo "---------------------------"
apt-get update
/opt/intel/computer_vision_sdk_2017.0.139/install_dependencies/install_cv_sdk_dependencies.sh
apt-get -y install build-essential cmake libopencv-dev checkinstall pkg-config yasm libjpeg-dev curl imagemagick gedit mplayer

# get the tutorial package
#echo "---------------------------"
#echo "downloading the tutorial package"
#echo "---------------------------"
#wget -O IE_tutorial_obj_recognition.tgz https://software.intel.com/file/594304/download
#tar -xzf IE_tutorial_obj_recognition.tgz


#-----------
# get CVSDK
echo "---------------------------"
echo "downloading Intel(R) Computer Vision SDK..."
echo "---------------------------"
curl -# -O http://registrationcenter-download.intel.com/akdlm/irc_nas/12361/intel_cv_sdk_ubuntu_r3_2017.0.139.tgz

#----------
# install CVSDK
echo "---------------------------"
echo "installing Intel(R) Computer Vision SDK..."
echo "---------------------------"
tar -xzf  intel_cv_sdk_ubuntu_r3_2017.0.139.tgz
cp silent.cfg intel_cv_sdk_ubuntu_r3_2017.0.139
cd intel_cv_sdk_ubuntu_r3_2017.0.139
./install.sh -s silent.cfg 
cd ..






