import sys,os
import os.path
import argparse


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


	#because script works in /opt directory it must be run as root
	if os.geteuid() != 0:
    		exit("This script must be run as root.  Please type 'sudo su' and run again.")


	#check if the patched Caffe exists
	if os.path.exists("/opt/intel/ssdcaffe"):
		print "Found mocaffe directory" 
	else:
		exit("Model Optimizer requires a patched Caffe to run.  Please run 'python installCaffe.py' first.")


	# get command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-w", help="path to model weights (.caffemodel file)",required=True)
        parser.add_argument("-d", help="path to model description (deploy .prototxt file) ", required=True)
	parser.add_argument("-mode", help="mode: CPU or GPU ", nargs=1, default="CPU")	
	args = parser.parse_args()
        

	#build a script to run MO from its bin directory
	cwd = os.getcwd()
	weightsfile=args.w
	if not weightsfile.find("/")==1: weightsfile=cwd+"/"+weightsfile
 
	descriptionfile=args.d
	if not descriptionfile.find("/")==1: descriptionfile=cwd+"/"+descriptionfile

	if ("GPU" in args.mode): precisionstr="FP16"
	else: precisionstr="FP32"

	cmd ="export FRAMEWORK_HOME=/opt/intel/ssdcaffe/caffe/build/lib;"

	if ("computer_vision_sdk" in IE_ROOT):
		cmd+="cd %s/mo/model_optimizer_caffe/bin;"%(IE_ROOT)
	else:
		cmd+="cd %s/model_optimizer/model_optimizer_caffe/bin;"%(IE_ROOT) 
        cmd+="./ModelOptimizer -i "
        cmd+=" -w  %s "%(weightsfile)
        cmd+=" -d  %s "%(descriptionfile)
        cmd+=" -f 1 "
        cmd+="-p %s "%(precisionstr)
        cmd+="--target XEON --network LOCALIZATION -b 1 "
        cmd+="-o %s "%(cwd+"/artifacts")

	print cmd
	os.system(cmd)



if __name__ == "__main__":
    main()
