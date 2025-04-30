#include "monocular-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <chrono>
#include <ranges>
#include <pthread.h>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
    : Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "/camera/rgb/image_color",
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

    RCLCPP_INFO(this->get_logger(), "Notify thread");

    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
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
                {
                    std::unique_lock<std::mutex> lock(syncMutex);
                    std::ranges::sort(keyFrameBuff, ORB_SLAM3::KeyFrame::lId);
                    lastKF = keyFrameBuff.back();
                    newDataReady = true;
                }
                processCV.notify_all();
                RCLCPP_INFO(this->get_logger(), "Detect keyFrame keyFrameId = %ld, frameId = %ld, size of bag %ld",
                            lastKF->mnId, lastKF->mnFrameId, keyFrameBuff.size());
            }
            prevSizeOfSet = keyFrameBuff.size();
        }
    }
}

void MonocularSlamNode::publishPoint(std::vector<geometry_msgs::msg::Point>&& vectToPublish, std_msgs::msg::ColorRGBA& color, geometry_msgs::msg::Vector3& scale)
{
    static unsigned long int counter = 0;
    std::unique_lock<std::mutex> lock(publishMutex);

    visualization_msgs::msg::Marker marker;
    // Задаем основные параметры маркера
    marker.header.frame_id = "camera_frame"; // Система координат
    marker.header.stamp = this->now();
    marker.ns = "points";
    marker.id = counter;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Размер точек
    marker.scale = scale;
    // Цвет (R, G, B, A)
    marker.color = color;
    marker.lifetime = rclcpp::Duration(0, 0);
    marker.points.insert(marker.points.begin(), vectToPublish.begin(), vectToPublish.end());

    objectPointPublisher->publish(marker);
    RCLCPP_INFO(this->get_logger(), "Publish points");
    counter++;
}

void MonocularSlamNode::visualizerLoop()
{

    std_msgs::msg::ColorRGBA color;
    color.a = 1.0f;
    color.b = 0.0f;
    color.r = 0.0f;
    color.g = 1.0f;

    geometry_msgs::msg::Vector3 scale;
    scale.x = 0.005;
    scale.y = 0.005;
    scale.z = 0.005;

    uint64_t prevSizeOfAllMapPoint = 0;
    while (rclcpp::ok() && ! stopThread)
    {
        std::vector<geometry_msgs::msg::Point> points;

        std::unique_lock<std::mutex> lock(syncMutex);
        processCV.wait(lock, [this]() {
            return newDataReady;
        });
        if (stopThread)
            break;
        auto activeMap = m_SLAM->mpAtlas->GetCurrentMap();
        const auto vpMPs = activeMap->GetAllMapPoints();

        if (vpMPs.size() == prevSizeOfAllMapPoint || vpMPs.empty())
            continue;
        // when changing map
        if (vpMPs.size() < prevSizeOfAllMapPoint)
            prevSizeOfAllMapPoint = 0;

        long int size = vpMPs.size() - prevSizeOfAllMapPoint;
        points.reserve(std::abs(size));

        auto view = std::ranges::drop_view {vpMPs, prevSizeOfAllMapPoint};
        for (const auto& it : view)
        {
            auto pose = it->GetWorldPos();
            geometry_msgs::msg::Point p1;
            p1.x = pose[0];
            p1.z = pose[1];
            p1.y = pose[2];
            points.push_back(p1);
        }
        // RCLCPP_INFO(this->get_logger(), "done");

        publishPoint(std::move(points), color, scale);

        newDataReady = false;
        prevSizeOfAllMapPoint = vpMPs.size();
    }
    RCLCPP_INFO(this->get_logger(), "visualizer threade done");
}

void MonocularSlamNode::imageLoop()
{
    std_msgs::msg::ColorRGBA color;
    color.a = 1.0f;
    color.b = 0.0f;
    color.r = 1.0f;
    color.g = 0.0f;

    geometry_msgs::msg::Vector3 scale;
    scale.x = 0.04;
    scale.y = 0.04;
    scale.z = 0.04;
    bool firstKF = true;
    while (rclcpp::ok() && ! stopThread)
    {
        std::unique_lock<std::mutex> lock(syncMutex);
        processCV.wait(lock, [this]() {
            return newDataReady;
        });
        if (stopThread)
            break;

        if (lastKF != nullptr)
        {
            if (firstKF)
            {
                publishStartPose(lastKF->GetPose().inverse());
                firstKF = false;
            }

            publishCameraPose(lastKF->GetPose().inverse());
            auto img = imgBuff.find_by_id(lastKF->mnFrameId);
            if (img.rows > 0)
            {
                auto objCoord = imgDtct.processImage(img, true);
                if (objCoord->size() != 0)
                {
                    RCLCPP_INFO(this->get_logger(), "Detect %ld object", objCoord->size());

                    std::vector<geometry_msgs::msg::Point> pointsVect;
                    for (const auto& coord : *objCoord)
                    {
                        std::vector<geometry_msgs::msg::Point> pointsOfObject;

                        RCLCPP_INFO(this->get_logger(), "Coord {%f,%f,%f} ", coord.x, coord.y, coord.r);
                        auto p = lastKF->GetFeaturesInArea(coord.x, coord.y, coord.r);
                        if (p.size() > 1)
                        {
                            for (const auto& point : p)
                            {
                                auto _mapPoint = lastKF->GetMapPoint(point);
                                if (_mapPoint)
                                {
                                    auto pose = _mapPoint->GetWorldPos();
                                    geometry_msgs::msg::Point p1;
                                    p1.x = pose[0];
                                    p1.z = pose[1];
                                    p1.y = pose[2];
                                    pointsOfObject.push_back(p1);
                                    if (pointsOfObject.size() > 10)
                                    {
                                        break;
                                    }
                                    RCLCPP_INFO(this->get_logger(), "Point coord {%f, %f, %f}", p1.x, p1.y, p1.z);
                                    pointsVect.push_back(p1);

                                }
                            }
                            // pointsVect.push_back(foundCentroid(pointsOfObject));
                        }
                    }

                    if (! pointsVect.empty() && pointsVect.size() < 200)
                    {
                        publishPoint(std::move(pointsVect), color, scale);
                    }
                }
                RCLCPP_INFO(this->get_logger(), "Image processed successfully");
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Empy image");
            }
        }
        newDataReady = false;
    }
    RCLCPP_INFO(this->get_logger(), "image detector threade done");
}

void MonocularSlamNode::publishCameraPose(Sophus::SE3f Tcw_SE3f)
{
    path.header.frame_id = "camera_frame";
    path.header.stamp = this->now();

    auto pose = geometry_msgs::msg::PoseStamped();
    pose.header = path.header;
    pose.pose.position.x = Tcw_SE3f.translation().x();
    pose.pose.position.z = Tcw_SE3f.translation().y();
    pose.pose.position.y = Tcw_SE3f.translation().z();
    pose.pose.orientation.w = Tcw_SE3f.unit_quaternion().coeffs().w();
    pose.pose.orientation.x = Tcw_SE3f.unit_quaternion().coeffs().x();
    pose.pose.orientation.y = Tcw_SE3f.unit_quaternion().coeffs().y();
    pose.pose.orientation.z = Tcw_SE3f.unit_quaternion().coeffs().z();
    path.poses.push_back(pose);

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

    frameId++;
    imgBuff.push(frameId, m_cvImPtr->image);
    m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));
    checkKF();
    // RCLCPP_INFO(this->get_logger(), "Frame counter %ld", frameId);
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
