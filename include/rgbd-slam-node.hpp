#pragma once
#include "monocular-slam-node.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class RGBD : public MonocularSlamNode
{
public:
    RGBD(ORB_SLAM3::System* pSLAM, bool _visualization);
    ~RGBD();
    void GrabRGBD(const sensor_msgs::msg::Image::SharedPtr msgRGB, const sensor_msgs::msg::Image::SharedPtr msgD);

private:
    // using ImageMsg = sensor_msgs::msg::Image;

    using RGBSub = message_filters::Subscriber<ImageMsg>;
    using DepthSub = message_filters::Subscriber<ImageMsg>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    // ORB_SLAM3::System* m_SLAM;

    RGBSub m_rgb_sub;
    DepthSub m_depth_sub;
    std::shared_ptr<Sync> m_sync;
    // ORB_SLAM3::System* m_SLAM;
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;
};
