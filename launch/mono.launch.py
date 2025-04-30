from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.actions import ExecuteProcess, DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    orbslam3_dir = get_package_share_directory("orbslam3")
    custom_lib_path = os.path.join(orbslam3_dir, "lib")

    SetEnvironmentVariable(name="LD_LIBRARY_PATH", value=f"{custom_lib_path}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    use_rviz = LaunchConfiguration("use_rviz")

    orb_slam3_node = Node(
        package="orbslam3",
        executable="mono",
        output=["screen"],
        arguments=[
            "/home/dima/diplom_ros/src/orbslam3_ros2/vocabulary/ORBvoc.txt",
            # "/home/dima/diplom_ros/src/orbslam3_ros2/config/monocular/TUM1.yaml",
            "/home/dima/diplom_ros/src/orbslam3_ros2/config/monocular/EuRoC.yaml",
        ],
    )

    # ros2 run usb_cam usb_cam_node_exe --ros-args --params-file ~/diplom_ros/src/orbslam3_ros2/config/camera_param.yaml
    # ros2 run v4l2_camera v4l2_camera_node --ros-args video_device:="/dev/video2"
    camera_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        output=["screen"],
        arguments=["params-file ~/diplom_ros/src/orbslam3_ros2/config/camera_param.yaml"],
    )
    # camera_node = Node(
    #     package="v4l2_camera",
    #     executable="v4l2_camera_node",
    #     output=["screen"],
    #     arguments=["video_device:='/dev/video2'"],
    # )

    sim = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            # "--loop",
            # "/home/dima/rosbag2_kitchen",
            #https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_floor
            "/home/dima/Desktop/V1_01_easy/freiburg2_desk",
        ],
        output="screen",
    )

    tf_static_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        name='map_to_base_link'
    )

    default_rviz_config_path = os.path.join(orbslam3_dir, "rviz/marker.rviz")
    print(default_rviz_config_path)
    rviz_node = Node(
        condition=IfCondition((use_rviz)),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config_path],
    )
    run_rviz_node = TimerAction(period=8.0, actions=[rviz_node])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="use_rviz",
                default_value="True",
                description="Start RViz",
            ),
            orb_slam3_node,
            sim,
            tf_static_node,
            run_rviz_node
            # rviz_node,
            # camera_node,
        ]
    )
