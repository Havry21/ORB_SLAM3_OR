#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <list>
#include "det/YOLO11.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <condition_variable>

struct ObjCoord
{
    float x;
    float y;
    float r;
};

class ImageDetector
{
public:
    ImageDetector() = default;
    ImageDetector(std::string modelName, bool _showDebugWindow = false, bool _saveImg = false);

    ~ImageDetector() = default;
    std::list<ObjCoord>* processImage(cv::Mat& image);
    void stopThread()
    {        
        stopThreads = true;
        cv.notify_all();
    };

private:
    void showAndSaveImage();
    std::string labelsPath = "/models/coco.names";
    std::string modelPath = "/models/yolo11l.onnx";
    bool isGPU = true;
    YOLO11Detector detector;
    std::list<ObjCoord> coordinate;

    std::thread visualThread;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopThreads = false;
    std::string outputDir = "";
    std::queue<cv::Mat> imageQueue;
    std::queue<std::vector<Detection>> resQueue;

    bool showDebugWindow = false;
    bool saveImg = false;
};
