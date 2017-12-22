/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <gflags/gflags.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <list>
#include <time.h>
#include <limits>
#include <chrono>

#include "common.hpp"
#include <ie_plugin_ptr.hpp>
#include <ie_plugin_config.hpp>
#include <ie_cnn_net_reader.h>
#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
using namespace cv;

using namespace InferenceEngine::details;
using namespace InferenceEngine;

#define DEFAULT_PATH_P "./lib"
#define NUMLABELS 20

#ifndef OS_LIB_FOLDER
#define OS_LIB_FOLDER "/"
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message";
/// @brief message for images argument
static const char image_message[] = "Required. Path to input video file";
/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";
/// @brief message for model argument
static const char model_message[] = "Required. Path to IR .xml file.";
/// @brief message for labels argument
static const char labels_message[] = "Required. Path to labels file.";
/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. (MKLDNNPlugin, clDNNPlugin) Force load specified plugin ";
/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Infer target device (CPU or GPU)";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";
/// @brief message for performance counters
static const char threshold_message[] = "confidence threshold for bounding boxes 0-1";
/// @brief message for batch size
static const char batch_message[] = "Batch size";
/// @brief message for frames count
static const char frames_message[] = "Number of frames from stream to process";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// \brief Define parameter for set video file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);
/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// \brief Define parameter for labels file <br>
/// It is a required parameter
DEFINE_string(l, "", labels_message);
/// \brief Define parameter for set plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// \brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, DEFAULT_PATH_P, plugin_path_message);
/// \brief device the target device to infer on <br>
DEFINE_string(d, "", target_device_message);
/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);
/// \brief Enable per-layer performance report
DEFINE_double(thresh, .4, threshold_message);
/// \brief Batch size
DEFINE_int32(batch, 1, batch_message);
/// \brief Frames count
DEFINE_int32(fr, -1, frames_message);

/**
 * \brief This function show a help message
 */
static void showUsage()
{
    std::cout << std::endl;
    std::cout << "IEclassification [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h           " << help_message << std::endl;
    std::cout << "    -i <path>    " << image_message << std::endl;
    std::cout << "    -fr <path>   " << frames_message << std::endl;
    std::cout << "    -m <path>    " << model_message << std::endl;
    std::cout << "    -l <path>    " << labels_message << std::endl;
    std::cout << "    -d <device>  " << target_device_message << std::endl;
    std::cout << "    -pc          " << performance_counter_message << std::endl;
    std::cout << "    -thresh <val>" << threshold_message << std::endl;
    std::cout << "    -b <val>     " << batch_message << std::endl;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float boxIntersection(DetectedObject a, DetectedObject b)
{
    float w = overlap(a.xmin, (a.xmax - a.xmin), b.xmin, (b.xmax - b.xmin));
    float h = overlap(a.ymin, (a.ymax - a.ymin), b.ymin, (b.ymax - b.ymin));

    if (w < 0 || h < 0) {
        return 0;
    }

    float area = w * h;

    return area;
}

float boxUnion(DetectedObject a, DetectedObject b)
{
    float i = boxIntersection(a, b);
    float u = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - i;

    return u;
}

float boxIoU(DetectedObject a, DetectedObject b)
{
    return boxIntersection(a, b) / boxUnion(a, b);
}

void doNMS(std::vector<DetectedObject>& objects, float thresh)
{
    int i, j;
    for(i = 0; i < objects.size(); ++i) {
        int any = 0;

        any = any || (objects[i].objectType > 0);

        if(!any) {
            continue;
        }

        for(j = i + 1; j < objects.size(); ++j) {
            if (boxIoU(objects[i], objects[j]) > thresh) {
                if (objects[i].prob < objects[j].prob) {
                    objects[i].prob = 0;
                }
                else {
                    objects[j].prob = 0;
                }
            }
        }
    }
}

/**
 * \brief This function analyses the YOLO net output for a single class
 * @param net_out - The output data
 * @param class_num - The class number
 * @return a list of found boxes
 */
std::vector < DetectedObject > yoloNetParseOutput(float *net_out,
        int class_num,
        int modelWidth,
        int modelHeight,
        float threshold
                                                 )
{
    int C = 20;         // classes
    int B = 2;          // bounding boxes
    int S = 7;          // cell size

    std::vector < DetectedObject > boxes;
    std::vector < DetectedObject > boxes_result;
    int SS = S * S;     // number of grid cells 7*7 = 49
    // First 980 values correspons to probabilities for each of the 20 classes for each grid cell.
    // These probabilities are conditioned on objects being present in each grid cell.
    int prob_size = SS * C; // class probabilities 49 * 20 = 980
    // The next 98 values are confidence scores for 2 bounding boxes predicted by each grid cells.
    int conf_size = SS * B; // 49*2 = 98 confidences for each grid cell

    float *probs = &net_out[0];
    float *confs = &net_out[prob_size];
    float *cords = &net_out[prob_size + conf_size]; // 98*4 = 392 coords x, y, w, h

    for (int grid = 0; grid < SS; grid++)
    {
        int row = grid / S;
        int col = grid % S;
        for (int b = 0; b < B; b++)
        {
            float conf = confs[(grid * B + b)];
            float prob = probs[grid * C + class_num] * conf;
            prob *= 3;  //TODO: probabilty is too low... check.

            if (prob < threshold) continue;

            float xc = (cords[(grid * B + b) * 4 + 0] + col) / S;
            float yc = (cords[(grid * B + b) * 4 + 1] + row) / S;
            float w = pow(cords[(grid * B + b) * 4 + 2], 2);
            float h = pow(cords[(grid * B + b) * 4 + 3], 2);

            DetectedObject bx(class_num,
                              (xc - w / 2)*modelWidth,
                              (yc - h / 2)*modelHeight,
                              (xc + w / 2)*modelWidth,
                              (yc + h / 2)*modelHeight,
                              prob);

            boxes_result.push_back(bx);
        }
    }

    return boxes_result;
}

/**
* \brief This function prints performance counters
* @param perfomanceMap - map of rerformance counters
*/
void printPerformanceCounters(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfomanceMap) {
    long long totalTime = 0;

    std::cout << std::endl << "Perfomance counts:" << std::endl << std::endl;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>::const_iterator it;

    for (it = perfomanceMap.begin(); it != perfomanceMap.end(); ++it) {
        std::cout << std::setw(30) << std::left << it->first + ":";
        switch (it->second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            std::cout << std::setw(15) << std::left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            std::cout << std::setw(15) << std::left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            std::cout << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }

        std::cout << std::setw(20) << std::left << "realTime: " + std::to_string(it->second.realTime_uSec);
        std::cout << std::setw(20) << std::left << " cpu: " + std::to_string(it->second.cpu_uSec);
        std::cout << std::endl;

        if (it->second.realTime_uSec > 0) {
            totalTime += it->second.realTime_uSec;
        }
    }

    std::cout << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " milliseconds" << std::endl;
}

/**
 * \brief The main function of inference engine sample application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {
    // ----------------
    // Parse command line parameters
    // ----------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    if (FLAGS_h) {
        showUsage();
        return 1;
    }

    if (FLAGS_l.empty()) {
        std::cout << "ERROR: labels file path not set" << std::endl;
        showUsage();
        return 1;
    }

    bool noPluginAndBadDevice = FLAGS_p.empty() && FLAGS_d.compare("CPU")
                                && FLAGS_d.compare("GPU");
    if (FLAGS_i.empty() || FLAGS_m.empty() || noPluginAndBadDevice) {
        if (noPluginAndBadDevice)
            std::cout << "ERROR: device is not supported" << std::endl;
        if (FLAGS_m.empty())
            std::cout << "ERROR: file with model - not set" << std::endl;
        if (FLAGS_i.empty())
            std::cout << "ERROR: image(s) for inference - not set" << std::endl;
        showUsage();
        return 2;
    }



    //----------------------------------------------------------------------------
    // prepare video input
    //----------------------------------------------------------------------------

    std::string input_filename = FLAGS_i;

    bool SINGLE_IMAGE_MODE = false;

    std::vector<std::string> imgExt = { ".bmp", ".jpg" };

    for (size_t i = 0; i < imgExt.size(); i++) {
        if (input_filename.rfind(imgExt[i]) != std::string::npos) {
            SINGLE_IMAGE_MODE = true;
            break;
        }
    }

    //open video capture
    VideoCapture cap(FLAGS_i.c_str());
    if (!cap.isOpened())   // check if VideoCapture init successful
    {
        std::cout << "Could not open input file" << std::endl;
        return 1;
    }

    // ----------------
    // Read class names
    // ----------------
    std::string labels[NUMLABELS];

    std::ifstream infile(FLAGS_l);

    if (!infile.is_open()) {
        std::cout << "Could not open labels file" << std::endl;
        return 1;
    }

    for (int i = 0; i < NUMLABELS; i++) {
        getline(infile, labels[i]);
    }

    infile.close();

    // -----------------
    // Load plugin
    // -----------------

#ifdef WIN32
    std::string archPath = "../../../bin" OS_LIB_FOLDER "intel64/Release/";
#else
    std::string archPath = "../../../lib/" OS_LIB_FOLDER "intel64";
#endif

    InferenceEngine::InferenceEnginePluginPtr _plugin(selectPlugin(
    {
        FLAGS_pp,
        archPath,
        DEFAULT_PATH_P,
        ""
        /* This means "search in default paths including LD_LIBRARY_PATH" */
    },
    FLAGS_p,
    FLAGS_d));

    const PluginVersion *pluginVersion;
    _plugin->GetVersion((const InferenceEngine::Version*&)pluginVersion);
    std::cout << pluginVersion << std::endl;

    // ----------------
    // Enable performance counters
    // ----------------

    if (FLAGS_pc) {
        _plugin->SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } }, nullptr);
    }

    // ----------------
    // Read network
    // ----------------
    InferenceEngine::CNNNetReader network;

    try {
        network.ReadNetwork(FLAGS_m);
    }
    catch (InferenceEngineException ex) {
        std::cerr << "Failed to load network: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "Network loaded." << std::endl;

    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    network.ReadWeights(binFileName.c_str());

    // ---------------
    // set the target device
    // ---------------
    if (!FLAGS_d.empty()) {
        network.getNetwork().setTargetDevice(getDeviceFromStr(FLAGS_d));
    }

    // --------------------
    // Set batch size
    // --------------------

    if (SINGLE_IMAGE_MODE) {
        network.getNetwork().setBatchSize(1);
    }
    else {
        network.getNetwork().setBatchSize(FLAGS_batch);
    }

    size_t batchSize = network.getNetwork().getBatchSize();

    std::cout << "Batch size = " << batchSize << std::endl;

    //----------------------------------------------------------------------------
    //  Inference engine input setup
    //----------------------------------------------------------------------------

    std::cout << "Setting-up input, output blobs..." << std::endl;

    // ---------------
    // set input configuration
    // ---------------
    InputsDataMap inputs;

    inputs = network.getNetwork().getInputsInfo();

    if (inputs.size() != 1) {
        std::cout << "This sample accepts networks having only one input." << std::endl;
        return 1;
    }

    InputInfo::Ptr ii = inputs.begin()->second;
    InferenceEngine::SizeVector inputDims = ii->getDims();

    if (inputDims.size() != 4) {
        std::cout << "Not supported input dimensions size, expected 4, got "
                  << inputDims.size() << std::endl;
    }

    std::string imageInputName = inputs.begin()->first;

    // look for the input == imageData
    DataPtr image = inputs[imageInputName]->getInputData();

    inputs[imageInputName]->setInputPrecision(FP32);

    // --------------------
    // Allocate input blobs
    // --------------------
    InferenceEngine::BlobMap inputBlobs;
    InferenceEngine::TBlob<float>::Ptr input =
        InferenceEngine::make_shared_blob < float,
        const InferenceEngine::SizeVector >(FP32, inputDims);
    input->allocate();

    inputBlobs[imageInputName] = input;

    //uncomment below for video output
    int frame_width = inputDims[1];
    int frame_height = inputDims[0];

    VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10,
                      Size(frame_width, frame_height), true);

    // --------------------------------------------------------------------------
    // Load model into plugin
    // --------------------------------------------------------------------------
    std::cout << "Loading model to plugin..." << std::endl;

    InferenceEngine::ResponseDesc dsc;

    InferenceEngine::StatusCode sts = _plugin->LoadNetwork(network.getNetwork(), &dsc);
    if (sts != 0) {
        std::cout << "Error loading model into plugin: " << dsc.msg << std::endl;
        return 1;
    }

    //----------------------------------------------------------------------------
    //  Inference engine output setup
    //----------------------------------------------------------------------------

    // --------------------
    // get output dimensions
    // --------------------
    InferenceEngine::OutputsDataMap out;
    out = network.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::string outputName = out.begin()->first;

    int maxProposalCount = -1;

    for (auto && item : out) {
        InferenceEngine::SizeVector outputDims = item.second->dims;

        InferenceEngine::TBlob < float >::Ptr output;
        output =
            InferenceEngine::make_shared_blob < float,
            const InferenceEngine::SizeVector >(InferenceEngine::FP32,
                                                outputDims);
        output->allocate();

        outputBlobs[item.first] = output;
        maxProposalCount = outputDims[1];

        std::cout << "maxProposalCount = " << maxProposalCount << std::endl;
    }

    InferenceEngine::SizeVector outputDims = outputBlobs.cbegin()->second->dims();
    size_t outputSize = outputBlobs.cbegin()->second->size() / batchSize;

    std::cout << "Output size = " << outputSize << std::endl;
    //----------------------------------------------------------------------------
    //---------------------------
    // main loop starts here
    //---------------------------
    //----------------------------------------------------------------------------

    Mat frame;
    Mat* resized = new Mat[batchSize];


    auto input_channels = inputDims[2];  // channels for color format.  RGB=4
    size_t input_width = inputDims[1];
    size_t input_height = inputDims[0];
    auto channel_size = input_width * input_height;
    auto input_size = channel_size * input_channels;

    std::cout << "Input size = " << input_size << std::endl;

    bool no_more_data = false;

    int totalFrames = 0;

    std::cout << "Running inference..." << std::endl;

    for (;;) {
        for (size_t mb = 0; mb < batchSize; mb++) {
            float* inputPtr = input.get()->data() + input_size * mb;

            //---------------------------
            // get a new frame
            //---------------------------
            cap >> frame;

            totalFrames++;

            if (!frame.data) {
                no_more_data = true;
                break;  //frame input ended
            }

            //---------------------------
            // resize to expected size (in model .xml file)
            //---------------------------
            resize(frame, resized[mb], Size(frame_width, frame_height));

            //---------------------------
            // PREPROCESS STAGE:
            // convert image to format expected by inference engine
            // IE expects planar, convert from packed
            //---------------------------
            long unsigned int framesize = resized[mb].rows * resized[mb].step1();

            if (framesize != input_size) {
                std::cout << "input pixels mismatch, expecting " << input_size
                          << " bytes, got: " << framesize;
                return 1;
            }

            // imgIdx = image pixel counter
            // channel_size = size of a channel, computed as image size in bytes divided by number of channels, or image width * image height
            // inputPtr = a pointer to pre-allocated inout buffer
            for (size_t i = 0, imgIdx = 0, idx = 0; i < channel_size; i++, idx++) {
                for (size_t ch = 0; ch < input_channels; ch++, imgIdx++) {
                    inputPtr[idx + ch * channel_size] = resized[mb].data[imgIdx];
                }
            }
        }

        if (no_more_data) {
            break;
        }

        if (FLAGS_fr > 0 && totalFrames > FLAGS_fr) {
            break;
        }

        //---------------------------
        // INFER STAGE
        //---------------------------
        sts = _plugin->Infer(inputBlobs, outputBlobs, &dsc);
        if (sts != 0) {
            std::cout << "An infer error occurred: " << dsc.msg << std::endl;
            return 1;
        }

        //---------------------------
        // Read perfomance counters
        //---------------------------

        if (FLAGS_pc) {
            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;
            _plugin->GetPerformanceCounts(perfomanceMap, nullptr);

            printPerformanceCounters(perfomanceMap);
        }

        //---------------------------
        // POSTPROCESS STAGE:
        // parse output
        // Output layout depends on network topology
        // so there are different paths for YOLO and SSD
        //---------------------------
        InferenceEngine::Blob::Ptr detectionOutBlob = outputBlobs[outputName];
        const InferenceEngine::TBlob < float >::Ptr detectionOutArray =
            std::dynamic_pointer_cast <InferenceEngine::TBlob <
            float >>(detectionOutBlob);

        for (size_t mb = 0; mb < batchSize; mb++) {
            float *box = detectionOutArray->data() + outputSize * mb;

            std::vector < DetectedObject > detectedObjects;

            //---------------------------
            // parse SSD output
            //---------------------------
            for (int c = 0; c < maxProposalCount; c++) {
                float image_id = box[c * 7 + 0];
                float label = box[c * 7 + 1] - 1;
                float confidence = box[c * 7 + 2];
                float xmin = box[c * 7 + 3] * inputDims[0];
                float ymin = box[c * 7 + 4] * inputDims[1];
                float xmax = box[c * 7 + 5] * inputDims[0];
                float ymax = box[c * 7 + 6] * inputDims[1];

                if (confidence > FLAGS_thresh) {
                    int labelnum=(int)label;
		    labelnum=(labelnum<0)?0:(labelnum>NUMLABELS)?NUMLABELS:labelnum;
                    printf("%d label=%s confidence %f (%f,%f)-(%f,%f)\n", c, labels[labelnum].c_str(), confidence, xmin, ymin, xmax, ymax);

                    rectangle(resized[mb],
                              Point((int)xmin, (int)ymin),
                              Point((int)xmax, (int)ymax), Scalar(0, 55, 255),
                              +1, 4);
                }
            }
        }

        //---------------------------
        // output/render
        //---------------------------

        for (int mb = 0; mb < batchSize; mb++) {
            if (SINGLE_IMAGE_MODE) {
                imwrite("out.jpg", resized[mb]);
            }
            else {
                video.write(resized[mb]);
            }

            //uncomment to render results
            //  cv::imshow("frame", resized[mb]);
            //  if (waitKey(30) >= 0)
            //      break;
        }
    }

    delete [] resized;

    std::cout << "Done!" << std::endl;

    return 0;
}
