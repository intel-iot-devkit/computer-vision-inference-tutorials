
cvsdk:
	g++ -O0 -g -o IEobjectdetection -std=c++11 main.cpp -I$(INTEL_CVSDK_DIR)/inference_engine/include -I./ -I$(INTEL_CVSDK_DIR)/inference_engine/samples/format_reader/  -I/usr/include/opencv -I/usr/local/include -I/opt/intel/computer_vision_sdk_2017.0.113/inference_engine/samples/thirdparty/gflags/include  -L$(INTEL_CVSDK_DIR)/inference_engine/bin/intel64/Release/lib -L$(INTEL_CVSDK_DIR)/inference_engine/lib/ubuntu_16.04/intel64 -ldl -linference_engine -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_video -lgflags_nothreads


  
dldt:
	g++ -O0 -g -o IEobjectdetection -std=c++11 main.cpp -I$(DLSDKROOT)/inference_engine/include -I./ -I$(DLSDKROOT)/inference_engine/samples/format_reader/  -I$(DLSDKROOT)/opencv/include/ -L./ -L $(DLSDKROOT)/opencv/lib/ -L$(DLSDKROOT)/inference_engine/bin/intel64/Release/lib/ -L$(DLSDKROOT)/inference_engine/lib/ubuntu_16.04/intel64/  -lgflags   -lpthread -ldl -linference_engine -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_video 

clean:
	rm -rf artifacts/ gflags-master gflags-master.zip  glog-master glog-master.zip IEobjectdetection out.avi out.jpg yolo_tiny/ yolo-tiny.tgz
  
