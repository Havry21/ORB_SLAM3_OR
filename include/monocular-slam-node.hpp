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

#include <utility>

#include "circulalBuffer.hpp"
#include "utility.hpp"
#include "image_inference.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "dbScan.hpp"

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM, bool _visualization);

    virtual ~MonocularSlamNode();

protected:
    using ImageMsg = sensor_msgs::msg::Image;
    using PointPub = rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr;

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void checkKF();
    void imageLoop();
    void visualizerLoop();
    void saveResultToYaml(const std::map<std::string, float>& data);

    void publishPoint(PointPub& pub, std::vector<geometry_msgs::msg::Point>& vectToPublish, std_msgs::msg::ColorRGBA& color, geometry_msgs::msg::Vector3& scale);
    void publishCameraPose(std::vector<geometry_msgs::msg::PoseStamped>& poses);
    void publishStartPose(Sophus::SE3f Tcw_SE3f);
    void publishCameraPoses();
    uint64_t calculateMedian(std::vector<uint64_t>& numbers);

    geometry_msgs::msg::Point foundCentroid(std::vector<geometry_msgs::msg::Point>& points);
    Clustering::DBSCAN dbScan = Clustering::DBSCAN(0.25, 2);
    ORB_SLAM3::System* m_SLAM;
    rclcpp::TimerBase::SharedPtr timer_;
    cv_bridge::CvImagePtr m_cvImPtr;
    std::vector<int> framdeId;
    int lastKFid = 0;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    PointPub objectPointPublisher;
    PointPub mapPointPublisher;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectoryPointPublisher;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tfStaticCamera;
    std::unordered_set<unsigned long> objectPointId;
    std::list<std::pair<long unsigned int, cv::Mat>> imgBuff;
    const size_t imgBufSize = 40;

    std::string package_dir;
    std::thread imageProcessThread;
    std::thread visualizerThread;

    std::condition_variable processCV;
    std::condition_variable processPub;

    std::mutex syncMutex;
    std::mutex publishMutex;
    std::mutex resMutex;
    bool visualization = false;
    bool saveImgs = false;
    bool newDataReady = false;
    bool stopThread = false;
    bool imgProccessComplete = true;
    std::string yoloName;
    ORB_SLAM3::KeyFrame* lastKF;
    std::vector<uint64_t> timePubPoints;

    std::vector<geometry_msgs::msg::Point> pointsMaps;
    std::vector<geometry_msgs::msg::Point> ObjectPointsUnsort;
    std::vector<geometry_msgs::msg::Point> ObjectPointsSort;

    size_t sizeOfPointsMaps;
    size_t sizeOfObjectPointsUnsort;
    size_t sizeOfObjectPointsSort;
};
