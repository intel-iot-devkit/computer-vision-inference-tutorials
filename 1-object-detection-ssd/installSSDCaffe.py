import sys,os
import os.path


#get a value for IE_ROOT from DLDT (DLSDKROOT) or CVSDK (INTEL_CVSDK_DIR)
#this script assumes that the DLDT setenv.sh script has been run with 'source' first.
def get_IE_ROOT():
	IE_ROOT="not_set"

	try:
		IE_ROOT=os.environ["DLSDKROOT"]
	except:
		print "DLSDKROOT not set (from Intel(R) Deep Learning Toolkit).  Trying INTEL_CVSDK_DIR."  
		

	try:
		IE_ROOT=os.environ["INTEL_CVSDK_DIR"]
	except:
		print "INTEL_CVSDK_DIR (from Intel(R) Computer Vision SDK) not set."  

        if ("not_set" in IE_ROOT):
		exit("Please source setenv.sh (from Intel(R) Deep Learning Toolkit) or setupvars.sh (from Intel(R) Computer Vision SDK).  Exiting.")

	return IE_ROOT




#main function
def main():

	# get value for IE_ROOT environment variable
	# if not set, script will exit here
	IE_ROOT=get_IE_ROOT()
	print "IE_ROOT=",IE_ROOT

	#because script is installing stuff it must be run as root
	if os.geteuid() != 0:
    		exit("This script must be run as root.  Please type 'sudo su' and run again.")

        
	#build caffe with MO extensions in /opt/intel/ssdcaffe if it is not already there
	if os.path.exists("/opt/intel/ssdcaffe"):
		print "ssdcaffe directory exists" 
		return

	print "Installing Ubuntu 16.04 prerequisites"
	cmd ="apt-get update;"
        cmd+="apt-get -y install git build-essential gcc gcc-multilib cmake libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev liblmdb-dev;"
	cmd+="apt-get install -y --no-install-recommends libboost-dev libboost-all-dev libboost-tools-dev;"
	os.system(cmd)


	print "Installing gflags"
	cmd = "rm -rf gflags-master; wget https://github.com/gflags/gflags/archive/master.zip -O gflags-master.zip;"
	cmd+= "unzip gflags-master.zip; cd gflags-master; "
        cmd+= "cmake -DCMAKE_CXX_FLAGS=-fPIC .; make -j 4; make install;"  
	os.system(cmd)


	print "Installing glog"
	cmd = "rm -rf glog-master; wget https://github.com/google/glog/archive/master.zip -O glog-master.zip;"
	cmd+= "unzip glog-master.zip; cd glog-master; "
        cmd+= "cmake -DCMAKE_CXX_FLAGS=-fPIC .; make -j 4; make install;"  
	os.system(cmd)


	print "Downloading caffe"
	cmd ="mkdir -p /opt/intel/ssdcaffe; cd /opt/intel/ssdcaffe;"
	cmd+="git clone https://github.com/weiliu89/caffe.git;"
	cmd+="cd caffe;"
	cmd+="git checkout ssd;"
        os.system(cmd)

        print "Patching Caffe* to support dynamic shapes"
        cmd= "patch /opt/intel/ssdcaffe/caffe/src/caffe/layers/detection_output_layer.cpp < ssd_caffe.patch;"
	os.system(cmd)


	print "Copy MO adapters to the Caffe source structure"
	if ("computer_vision_sdk" in IE_ROOT):
	        cmd ="cp  %s/mo/model_optimizer_caffe/adapters/include/caffe/* /opt/intel/ssdcaffe/caffe/include/caffe/;"%(IE_ROOT)
        	cmd+="cp  %s/mo/model_optimizer_caffe/adapters/src/caffe/*     /opt/intel/ssdcaffe/caffe/src/caffe/;"%(IE_ROOT)
	else:
        	cmd ="cp  %s/model_optimizer/model_optimizer_caffe/adapters/include/caffe/* /opt/intel/ssdcaffe/caffe/include/caffe/;"%(IE_ROOT)
        	cmd+="cp  %s/model_optimizer/model_optimizer_caffe//adapters/src/caffe/*     /opt/intel/ssdcaffe/caffe/src/caffe/;"%(IE_ROOT)
	os.system(cmd)

	print "Set HDF5 environment"

	os.environ["PATH"] += os.pathsep + "/usr/lib/x86_64-linux-gnu/hdf5/serial"

	cmd = "cd /usr/lib/x86_64-linux-gnu;"
	cmd+= "rm libhdf5.so;"
	cmd+= "rm libhdf5_hl.so;"
	cmd+= "ln -s libhdf5_serial.so.10.1.0 libhdf5.so;"
	cmd+= "ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so;"
	os.system(cmd)

	print "Build Caffe"
        cmd ="cd /opt/intel/ssdcaffe/caffe; rm -rf build; mkdir build; cd build;"
	cmd+="cmake -DCPU_ONLY=1 -DUSE_OPENCV=0 -DBUILD_python=0 ..;"
        cmd+="make -j 4;"
	os.system(cmd)



if __name__ == "__main__":
    main()
