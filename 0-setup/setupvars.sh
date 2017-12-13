# this is a template file for script to setup environment variables
# the variable INSTALLDIR will contain full path to installation location of
# the suite

INSTALLDIR=/opt/intel/computer_vision_sdk_2017.1.163
export INTEL_CVSDK_DIR=$INSTALLDIR
if [ -e $INSTALLDIR/openvx ]; then
   export LD_LIBRARY_PATH=$INSTALLDIR/openvx/lib:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/mo ]; then
   export LD_LIBRARY_PATH=$INSTALLDIR/mo/model_optimizer_caffe/bin:/opt/intel/opencl:$LD_LIBRARY_PATH
fi

if [[ -f /etc/centos-release ]]; then
    IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/inference_engine/lib/centos_7.3/intel64
elif [[ -f /etc/lsb-release ]]; then
    UBUNTU_VERSION=$(lsb_release -r -s)
    if [[ $UBUNTU_VERSION = "16.04" ]]; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/inference_engine/lib/ubuntu_16.04/intel64
    elif [[ $UBUNTU_VERSION = "14.04" ]]; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/inference_engine/lib/ubuntu_14.04/intel64
    elif cat /etc/lsb-release | grep -q "Yocto" ; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/inference_engine/lib/ubuntu_16.04/intel64
    fi
fi

if [ -e $INSTALLDIR/inference_engine ]; then
   export LD_LIBRARY_PATH=$INSTALLDIR/inference_engine/external/cldnn/lib:$INSTALLDIR/inference_engine/external/mklml_lnx/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/opencv ]; then
   export OpenCV_DIR=$INSTALLDIR/opencv/share/OpenCV
   export LD_LIBRARY_PATH=$INSTALLDIR/opencv/lib:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=$INSTALLDIR/opencv/share/OpenCV/3rdparty/lib:$LD_LIBRARY_PATH
fi


