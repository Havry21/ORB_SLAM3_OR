#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <list>
#include "det/YOLO11.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
struct ObjCoord
{
    float x;
    float y;
    float r;
};

class ImageDetector
{
public:
    ImageDetector();
    ~ImageDetector() = default;
    std::list<ObjCoord>* processImage(cv::Mat& image, bool showDebugWindow = false);

private:
    std::string labelsPath = "/models/coco.names";
    std::string modelPath = "/models/yolo11n.onnx";
    bool isGPU = true;
    YOLO11Detector detector;
    std::list<ObjCoord> coordinate;
};
