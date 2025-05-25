

#include "image_inference.hpp"
ImageDetector::ImageDetector(std::string modelName, bool _showDebugWindow, bool _saveImg)
{
    std::string package_dir = ament_index_cpp::get_package_share_directory("orbslam3");
    outputDir += package_dir + "/result/img";
    labelsPath = package_dir + labelsPath;
    modelPath = package_dir + "/models/" + modelName + ".onnx";
    detector = YOLO11Detector(modelPath, labelsPath, false);
    showDebugWindow = _showDebugWindow;
    saveImg = _saveImg;
    if (showDebugWindow || saveImg)
    {
        visualThread = std::thread(&ImageDetector::showAndSaveImage, this);
        visualThread.detach();
    }
}

void ImageDetector::showAndSaveImage()
{
    int frameCount = 0;
    cv::Mat frame;
    std::vector<Detection> res;
    while (! stopThreads)
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return ! imageQueue.empty() || stopThreads; });

            if (stopThreads)
                break;

            frame = imageQueue.front();
            imageQueue.pop();
            res = resQueue.front();
            resQueue.pop();
        }

        detector.drawBoundingBox(frame, res);
        if (showDebugWindow)
        {
            cv::imshow("Detections", frame);
            cv::waitKey(1);
        }

        if (saveImg)
        {
            std::string filename = outputDir + "/frame_" + std::to_string(frameCount++) + ".jpg";
            cv::imwrite(filename, frame);
        }

    }
    std::cout << "Thread OpenCV done" << std::endl;
}

std::list<ObjCoord>* ImageDetector::processImage(cv::Mat& image)
{
    coordinate.clear();

    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image!\n";
        return &coordinate;
    }

    // Detect objects in the image and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> results = detector.detect(image, 41);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    // std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
    if (results.size() != 0 && (showDebugWindow || saveImg))
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            imageQueue.push(image.clone());
            resQueue.push(results);
        }
        cv.notify_all();
    }
    for (const auto& res : results)
    {
        int x = res.box.x + res.box.width / 2;
        int y = res.box.y + res.box.height / 2;
        int r = std::min(res.box.width, res.box.height) / 2;
        coordinate.push_back({.x = x, .y = y, .rx = res.box.width / 2, .ry = res.box.height / 2});
    }
    return &coordinate;
}
