#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "nav_msgs/msg/path.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "tf2_ros/static_transform_broadcaster.h"
#include <cv_bridge/cv_bridge.hpp>
#include <unordered_map>
#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "circulalBuffer.hpp"
#include "utility.hpp"
#include "image_inference.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);

    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void checkKF();
    void imageLoop();
    void visualizerLoop();

    void publishPoint(std::vector<geometry_msgs::msg::Point>&& vectToPublish, std_msgs::msg::ColorRGBA& color, geometry_msgs::msg::Vector3& scale);
    void publishCameraPose(Sophus::SE3f Tcw_SE3f);
    void publishStartPose(Sophus::SE3f Tcw_SE3f);

    geometry_msgs::msg::Point foundCentroid(std::vector<geometry_msgs::msg::Point>& points);

    ORB_SLAM3::System* m_SLAM;
    rclcpp::TimerBase::SharedPtr timer_;
    cv_bridge::CvImagePtr m_cvImPtr;
    std::vector<int> framdeId;
    int lastKFid = 0;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr objectPointPublisher;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mapPointPublisher;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectoryPointPublisher;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tfStaticCamera;

    nav_msgs::msg::Path path;
    CircularBuffer<10> imgBuff;

    std::thread imageProcessThread;
    std::thread visualizerThread;

    std::condition_variable processCV;
    std::mutex syncMutex;
    std::mutex publishMutex;

    bool newDataReady = false;
    bool stopThread = false;
    ImageDetector imgDtct;
    ORB_SLAM3::KeyFrame* lastKF;
};
