all:
	g++ -O0 -g -o IEobjectdetection -std=c++11 main.cpp -I$(INTEL_CVSDK_DIR)/inference_engine/include -I./ -I$(INTEL_CVSDK_DIR)/inference_engine/samples/format_reader/  -I$(INTEL_CVSDK_DIR)/opencv/include -I/usr/local/include -I$(INTEL_CVSDK_DIR)/inference_engine/samples/thirdparty/gflags/include  -L$(INTEL_CVSDK_DIR)/inference_engine/bin/intel64/Release/lib -L$(INTEL_CVSDK_DIR)/inference_engine/lib/ubuntu_16.04/intel64 -L$(INTEL_CVSDK_DIR)/opencv/lib -ldl -linference_engine -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lgflags_nothreads -lopencv_imgcodecs -lopencv_imgcodecs



