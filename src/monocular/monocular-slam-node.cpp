#include "monocular-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <chrono>
#include <ranges>
#include <pthread.h>
#include <yaml-cpp/yaml.h>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM, bool _visualization)
    : Node("ORB_SLAM3_ROS2")
{
    package_dir = ament_index_cpp::get_package_share_directory("orbslam3");

    this->declare_parameter<bool>("use_visualization", false);
    this->declare_parameter<bool>("save_imgs", false);

    this->declare_parameter<std::string>("yolo_model", "yolo11n");

    yoloName = this->get_parameter("yolo_model").as_string();
    visualization = this->get_parameter("use_visualization").as_bool();
    saveImgs = this->get_parameter("save_imgs").as_bool();

    RCLCPP_INFO(this->get_logger(), "Yolo version %s", yoloName.data());
    RCLCPP_INFO(this->get_logger(), "Visualization %s", (visualization ? "Enable" : "Disable"));
    RCLCPP_INFO(this->get_logger(), "Save img %s", (saveImgs ? "Enable" : "Disable"));

    m_SLAM = pSLAM;
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "/cam0/image_raw",
        // "/camera/rgb/image_color",
        // "/image_raw",
        10,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));

    objectPointPublisher = this->create_publisher<visualization_msgs::msg::Marker>("object_markers", 10);
    mapPointPublisher = this->create_publisher<visualization_msgs::msg::Marker>("map_markers", 10);
    trajectoryPointPublisher = this->create_publisher<nav_msgs::msg::Path>("trajectory_markers", 10);
    tfStaticCamera = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    imageProcessThread = std::thread(&MonocularSlamNode::imageLoop, this);
    visualizerThread = std::thread(&MonocularSlamNode::visualizerLoop, this);
    pthread_setname_np(imageProcessThread.native_handle(), "imageDetectorBackground");
    imageProcessThread.detach();
    pthread_setname_np(visualizerThread.native_handle(), "visualizerBackground");
    visualizerThread.detach();

    RCLCPP_INFO(this->get_logger(), "Node init");
}

MonocularSlamNode::~MonocularSlamNode()
{
    RCLCPP_INFO(this->get_logger(), "Node destroyed!");
    newDataReady = true;
    stopThread = true;

    processCV.notify_all();
    processPub.notify_all();
    RCLCPP_INFO(this->get_logger(), "Notify thread");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM({package_dir + "/result/" + "KeyFrameTrajectory.txt"});
}

void MonocularSlamNode::checkKF()
{
    static unsigned int prevkfId = 0;
    static unsigned int prevSizeOfSet = 0;

    auto _atlas = m_SLAM->mpAtlas;
    if (_atlas)
    {
        auto keyFrameBuff = _atlas->GetAllKeyFrames();
        if (! keyFrameBuff.empty())
        {
            // Либо добавился, либо удалился кадр
            if (prevSizeOfSet < keyFrameBuff.size())
            {
                // if (! imgProccessComplete)
                // {
                //     RCLCPP_INFO(this->get_logger(), "Pass keyFrame");
                // }
                // else
                // {
                {
                    std::unique_lock<std::mutex> lock(syncMutex);
                    std::ranges::sort(keyFrameBuff, ORB_SLAM3::KeyFrame::lId);
                    lastKF = keyFrameBuff.back();
                    newDataReady = true;
                }

                if (! imgProccessComplete)
                {
                    RCLCPP_INFO(this->get_logger(), "Pass keyFrame");
                }
                else
                {
                    processCV.notify_all();
                }

                processPub.notify_all();

                RCLCPP_INFO(this->get_logger(), "Detect keyFrame keyFrameId = %ld, frameId = %ld, size of bag %ld",
                            lastKF->mnId, lastKF->mnFrameId, keyFrameBuff.size());
                // }
            }
            prevSizeOfSet = keyFrameBuff.size();
        }
    }
}

void MonocularSlamNode::publishPoint(PointPub& pub, std::vector<geometry_msgs::msg::Point>& vectToPublish, std_msgs::msg::ColorRGBA& color, geometry_msgs::msg::Vector3& scale)
{
    std::unique_lock<std::mutex> lock(publishMutex);
    visualization_msgs::msg::Marker marker;
    // Задаем основные параметры маркера
    marker.header.frame_id = "camera_frame"; // Система координат
    marker.header.stamp = this->now();
    marker.ns = "points";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.scale = scale;
    // Цвет (R, G, B, A)
    marker.color = color;
    marker.lifetime = rclcpp::Duration(0, 0);
    marker.points.insert(marker.points.begin(), vectToPublish.begin(), vectToPublish.end());

    pub->publish(marker);
    RCLCPP_INFO(this->get_logger(), "Publish points");
}

void MonocularSlamNode::visualizerLoop()
{
    std_msgs::msg::ColorRGBA colorMaps;
    colorMaps.a = 1.0f;
    colorMaps.b = 0.0f;
    colorMaps.r = 0.0f;
    colorMaps.g = 1.0f;

    geometry_msgs::msg::Vector3 scaleMaps;
    scaleMaps.x = 0.005;
    scaleMaps.y = 0.005;
    scaleMaps.z = 0.005;

    std_msgs::msg::ColorRGBA colorObjects;
    colorObjects.a = 1.0f;
    colorObjects.b = 0.0f;
    colorObjects.r = 1.0f;
    colorObjects.g = 0.0f;

    geometry_msgs::msg::Vector3 scaleObjects;
    scaleObjects.x = 0.04;
    scaleObjects.y = 0.04;
    scaleObjects.z = 0.04;

    while (rclcpp::ok() && ! stopThread)
    {
        {
            std::unique_lock<std::mutex> lock(syncMutex);
            processPub.wait(lock, [this]() {
                return newDataReady;
            });
            if (stopThread)
                break;
        }
        auto startTime = std::chrono::steady_clock::now();

        const auto maps = m_SLAM->mpAtlas->GetAllMaps();
        for (const auto& map : maps)
        {
            const auto vpMPs = map->GetAllMapPoints();
            if (vpMPs.empty())
                continue;

            for (const auto& it : vpMPs)
            {
                auto pose = it->GetWorldPos();
                geometry_msgs::msg::Point p1;
                p1.x = pose[0];
                p1.z = pose[1];
                p1.y = pose[2];
                if (std::ranges::find(objectPointId, it->mnId) != objectPointId.end())
                {
                    ObjectPointsUnsort.push_back(p1);
                    if (ObjectPointsUnsort.size() < 3)
                    {
                        RCLCPP_INFO(this->get_logger(), "Point coord %.2f,%.2f,%.2f",
                                    p1.x, p1.z, p1.y);
                    }
                }
                {
                    pointsMaps.push_back(p1);
                }
            }
        }

        if (ObjectPointsUnsort.size() > 1)
        {
            RCLCPP_INFO(this->get_logger(), "Num of unsort point %ld", ObjectPointsUnsort.size());
            try
            {
                // RCLCPP_INFO(this->get_logger(), "Prepare clastering, size of obj %ld",ObjectPointsUnsort.size() );
                auto result = dbScan.cluster(ObjectPointsUnsort);
                RCLCPP_INFO(this->get_logger(), "end clastering");
                RCLCPP_INFO(this->get_logger(), "Num of objects %ld",
                            result.clusters.size());
                RCLCPP_INFO(this->get_logger(), "Prepare add in claster");
                for (auto& points : result.clusters)
                {
                    ObjectPointsSort.push_back(foundCentroid(points));
                }
            } catch (const char* error_message)
            {
                RCLCPP_ERROR(this->get_logger(), "Error in dbscan %s", error_message);
                // std::cout << error_message << std::endl;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Prepare to publish");
        try
        {
            publishCameraPoses();
            publishPoint(mapPointPublisher, pointsMaps, colorMaps, scaleMaps);
            publishPoint(objectPointPublisher, ObjectPointsSort, colorObjects, scaleObjects);
            publishCameraPoses();
        } catch (const char* error_message)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in publish point %s", error_message);
            // std::cout << error_message << std::endl;
        }
        sizeOfPointsMaps = pointsMaps.size();
        sizeOfObjectPointsUnsort = ObjectPointsSort.size();
        sizeOfObjectPointsSort = ObjectPointsUnsort.size();

        pointsMaps.clear();
        ObjectPointsSort.clear();
        ObjectPointsUnsort.clear();

        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
        timePubPoints.push_back(dt);

        newDataReady = false;
    }
    RCLCPP_INFO(this->get_logger(), "visualizer threade done");
    {
        std::unique_lock<std::mutex> lock(resMutex);
        RCLCPP_INFO(this->get_logger(), "Result \n\r Median postImg %ld ms", this->calculateMedian(timePubPoints));
    }
}

void MonocularSlamNode::publishCameraPoses()
{
    auto activeMap = m_SLAM->mpAtlas->GetCurrentMap();
    auto keyFrames = activeMap->GetAllKeyFrames();
    std::ranges::sort(keyFrames, ORB_SLAM3::KeyFrame::lId);

    std::vector<geometry_msgs::msg::PoseStamped> poses;
    geometry_msgs::msg::PoseStamped pose;

    for (const auto& kf : keyFrames)
    {
        auto Tcw_SE3f = kf->GetPose().inverse();
        pose.pose.position.x = Tcw_SE3f.translation().x();
        pose.pose.position.z = Tcw_SE3f.translation().y();
        pose.pose.position.y = Tcw_SE3f.translation().z();
        pose.pose.orientation.w = Tcw_SE3f.unit_quaternion().coeffs().w();
        pose.pose.orientation.x = Tcw_SE3f.unit_quaternion().coeffs().x();
        pose.pose.orientation.y = Tcw_SE3f.unit_quaternion().coeffs().y();
        pose.pose.orientation.z = Tcw_SE3f.unit_quaternion().coeffs().z();
        poses.push_back(pose);
    }
    publishCameraPose(poses);
}

uint64_t MonocularSlamNode::calculateMedian(std::vector<uint64_t>& numbers)
{
    std::sort(numbers.begin(), numbers.end());

    size_t size = numbers.size();
    size_t middle = size / 2;

    if (size % 2 == 0)
    {
        return (numbers[middle - 1] + numbers[middle]) / 2.0;
    }
    else
    {
        return numbers[middle];
    }
}

void MonocularSlamNode::imageLoop()
{
    ImageDetector imgDtct(yoloName, visualization, saveImgs);

    bool firstKF = true;
    std::vector<uint64_t> timeDecodeImg;
    std::vector<uint64_t> timePeriodKF;

    auto startPeriodKF = std::chrono::steady_clock::now();

    while (rclcpp::ok() && ! stopThread)
    {
        ORB_SLAM3::KeyFrame* _lastKF;
        {
            std::unique_lock<std::mutex> lock(syncMutex);
            processCV.wait(lock, [this]() {
                return newDataReady;
            });
            if (stopThread)
                break;
            imgProccessComplete = false;
            _lastKF = lastKF;
        }
        if (startPeriodKF != std::chrono::steady_clock::now())
        {
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startPeriodKF).count();
            timePeriodKF.push_back(dt);
        }

        if (firstKF)
        {
            publishStartPose(_lastKF->GetPose().inverse());
            firstKF = false;
        }
        cv::Mat img;
        {
            std::unique_lock<std::mutex> lock(syncMutex);
            std::string idStr;
            for (auto& buffElement : imgBuff)
            {
                idStr += " " + std::to_string(buffElement.first);
                if (buffElement.first == _lastKF->mnFrameId)
                {
                    img = buffElement.second;
                    break;
                }
            }
            RCLCPP_DEBUG(this->get_logger(), "Looks ids %ld", _lastKF->mnFrameId);
            RCLCPP_DEBUG(this->get_logger(), "Img ids %s", idStr.data());
        }
        if (! img.empty())
        {
            try
            {
                auto startDecodeImgTime = std::chrono::steady_clock::now();

                auto objCoord = imgDtct.processImage(img);
                auto imgProccesTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDecodeImgTime).count();
                if (! objCoord->empty())
                {
                    RCLCPP_INFO(this->get_logger(), "Detect %ld object", objCoord->size());
                    for (const auto& coord : *objCoord)
                    {
                        RCLCPP_INFO(this->get_logger(), "Coord {%f,%f,%f,%f} ", coord.x, coord.y, coord.rx, coord.ry);
                        // coord.r * 0.7: because there may be a point on the edges that is not visible to the object.
                        // auto p = _lastKF->GetFeaturesInArea(coord.x, coord.y, coord.rx * 0.7, coord.ry * 0.7);

                        auto p = _lastKF->GetFeaturesInSquare(coord.x, coord.y, coord.rx * 0.7, coord.ry * 0.7);
                        if (! p.empty())
                        {
                            for (const auto& point : p)
                            {
                                auto _mapPoint = _lastKF->GetMapPoint(point);
                                if (_mapPoint && ! _mapPoint->isBad())
                                {
                                    objectPointId.insert(_mapPoint->mnId);
                                }
                            }
                        }
                    }
                }
                auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDecodeImgTime).count();
                timeDecodeImg.push_back(dt);
                RCLCPP_INFO(this->get_logger(), "Image processed successfully, \n for %ld ms, total %ld ms", imgProccesTime, dt);
                RCLCPP_INFO(this->get_logger(), "Size of objectPointId %ld", objectPointId.size());
            } catch (const char* error_message)
            {
                RCLCPP_ERROR(this->get_logger(), "Error %s", error_message);
                // std::cout << error_message << std::endl;
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Empty image");
            RCLCPP_DEBUG(this->get_logger(), "Img don't found, frame id = %ld,ketFrame id = %ld", _lastKF->mnFrameId, _lastKF->mnId);
        }

        startPeriodKF = std::chrono::steady_clock::now();
        newDataReady = false;
        imgProccessComplete = true;
    }
    imgDtct.stopThread();

    auto medPeriodKF = this->calculateMedian(timePeriodKF);
    auto medProcessImg = this->calculateMedian(timeDecodeImg);

    RCLCPP_INFO(this->get_logger(), "Result \n\r Median perido of KF %ld ms, median processImage %ld ms",
                medPeriodKF, medProcessImg);
    float kfCompletion = 0.0f;
    {
        std::unique_lock<std::mutex> lock(resMutex);
        auto med = std::max(medProcessImg, this->calculateMedian(timePubPoints));
        kfCompletion = (float) med / (float) medPeriodKF;
        RCLCPP_INFO(this->get_logger(), "Result \n\r Percentage of keyframe completion %.5f", kfCompletion);
    }

    std::map<std::string, float> resultMap = {
        {"medPeriodKF", medPeriodKF},
        {"medProcessImg", medProcessImg},
        {"kfCompletion", kfCompletion},
        {"numOfMapPoints", sizeOfPointsMaps},
        {"numOfObjPoints", sizeOfObjectPointsUnsort},
        {"numOfFilteredObjPoints", sizeOfObjectPointsSort},
        {"thrh", CONFIDENCE_THRESHOLD}};
    saveResultToYaml(resultMap);

    RCLCPP_INFO(this->get_logger(), "image detector threade done");
}

void MonocularSlamNode::publishCameraPose(std::vector<geometry_msgs::msg::PoseStamped>& poses)
{
    nav_msgs::msg::Path path;
    path.header.frame_id = "camera_frame";
    path.header.stamp = this->now();

    for (auto& p : poses)
    {
        path.poses.push_back(p);
    }
    trajectoryPointPublisher->publish(path);
}

void MonocularSlamNode::publishStartPose(Sophus::SE3f Tcw_SE3f)
{
    geometry_msgs::msg::TransformStamped transformStamped;

    transformStamped.header.stamp = this->now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "camera_frame";

    transformStamped.transform.translation.x = Tcw_SE3f.translation().x();
    transformStamped.transform.translation.y = Tcw_SE3f.translation().y();
    transformStamped.transform.translation.z = Tcw_SE3f.translation().z();
    transformStamped.transform.rotation.x = Tcw_SE3f.unit_quaternion().coeffs().w();
    transformStamped.transform.rotation.y = Tcw_SE3f.unit_quaternion().coeffs().x();
    transformStamped.transform.rotation.z = Tcw_SE3f.unit_quaternion().coeffs().y();
    transformStamped.transform.rotation.w = Tcw_SE3f.unit_quaternion().coeffs().z();

    tfStaticCamera->sendTransform(transformStamped);
    RCLCPP_INFO(this->get_logger(), "Published static transform from 'map' to 'camera_frame'");
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    try
    {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    } catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    static unsigned long frameId = 0;
    static unsigned long counter = 0;

    // counter++;
    // if (counter % 5 == 0)
    // {
    //     return;
    // }
    frameId++;
    {
        std::unique_lock<std::mutex> lock(syncMutex);
        imgBuff.push_back({frameId, m_cvImPtr->image});
    }
    if (m_cvImPtr->image.rows <= 0)
    {
        RCLCPP_ERROR(this->get_logger(), "empty img");
    }
    m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));
    RCLCPP_DEBUG(this->get_logger(), "Pub img %ld", frameId);
    checkKF();

    {
        std::unique_lock<std::mutex> lock(syncMutex);

        if (imgBufSize < imgBuff.size())
        {
            imgBuff.pop_front();
        }
    }
}

geometry_msgs::msg::Point MonocularSlamNode::foundCentroid(std::vector<geometry_msgs::msg::Point>& points)
{
    geometry_msgs::msg::Point centroid;
    for (const auto& point : points)
    {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }

    auto size = points.size();

    centroid.x /= size;
    centroid.y /= size;
    centroid.z /= size;

    return centroid;
}

void MonocularSlamNode::saveResultToYaml(const std::map<std::string, float>& data)
{
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;

    for (const auto& pair : data)
    {
        emitter << YAML::Key << pair.first;
        emitter << YAML::Value << pair.second;
    }

    emitter << YAML::EndMap;
    auto fileName = package_dir + "/result/res.yaml";
    std::ofstream fout(fileName);
    fout << emitter.c_str();
    fout.close();
}
