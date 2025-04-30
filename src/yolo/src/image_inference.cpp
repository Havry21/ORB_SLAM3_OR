

#include "image_inference.hpp"
ImageDetector::ImageDetector()
{
    std::string package_dir = ament_index_cpp::get_package_share_directory("orbslam3");
    labelsPath = package_dir + labelsPath;
    modelPath = package_dir + modelPath;
    detector = YOLO11Detector(modelPath, labelsPath);
}

std::list<ObjCoord>* ImageDetector::processImage(cv::Mat& image, bool showDebugWindow)
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

    std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
    if (showDebugWindow)
    {
        detector.drawBoundingBox(image, results);
        cv::imshow("Detections", image);
    }
    for (const auto& res : results)
    {
        int x = res.box.x + res.box.width / 2;
        int y = res.box.y + res.box.height / 2;
        int r = std::min(res.box.width, res.box.height) / 2;
        coordinate.push_back({.x=x, .y=y, .r=r});
    }
    return &coordinate;
}
