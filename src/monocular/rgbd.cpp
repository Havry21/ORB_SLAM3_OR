#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rgbd-slam-node.hpp"

#include "System.h"


int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cerr << "\nUsage: ros2 run orbslam mono path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    rclcpp::init(argc, argv);

    // malloc error using new.. try shared ptr
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    bool visualization = std::strcmp("true",argv[3]) != 0  ? false : true;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, visualization);

    auto node = std::make_shared<RGBD>(&SLAM,visualization);
    std::cout << "============================ " << std::endl;\

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
