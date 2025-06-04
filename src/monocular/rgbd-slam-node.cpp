#include "rgbd-slam-node.hpp"

RGBD::RGBD(ORB_SLAM3::System* pSLAM, bool _visualization)
    : MonocularSlamNode(pSLAM, _visualization)
{
    m_rgb_sub.subscribe(this, "/camera/camera/rgb");
    m_depth_sub.subscribe(this, "/camera/camera/depth");

    m_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), m_rgb_sub, m_depth_sub);
    m_sync->setAgePenalty(0.50);
    m_sync->registerCallback(&RGBD::GrabRGBD, this);
    RCLCPP_INFO(this->get_logger(), "RGBD created");
}

RGBD::~RGBD() {

};

void RGBD::GrabRGBD(const sensor_msgs::msg::Image::SharedPtr msgRGB, const sensor_msgs::msg::Image::SharedPtr msgD)
{
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    } catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    } catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    static unsigned long frameId = 0;
    static unsigned long counter = 0;
    frameId++;
    {
        std::unique_lock<std::mutex> lock(syncMutex);
        imgBuff.push_back({frameId, cv_ptrRGB->image.clone()});
    }
    if (cv_ptrRGB->image.rows <= 0)
    {
        RCLCPP_ERROR(this->get_logger(), "empty img");
    }

    m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, Utility::StampToSec(msgRGB->header.stamp));

    checkKF();

    {
        std::unique_lock<std::mutex> lock(syncMutex);

        if (imgBufSize < imgBuff.size())
        {
            imgBuff.pop_front();
        }
    }
}