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



    sim = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "-r 5.0",
            # "--remap",
            # "/cam0/image_raw:=/camera/rgb/image_color",
            "/media/dima/additional/dataset/office6_RGBD1_point_final",
        ],
        output="screen",
    )
    
    tf_static_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '3.141592653589793', 'map', 'camera_frame'],
        name='map_to_base_link'
    )


    default_rviz_config_path = os.path.join(orbslam3_dir, "rviz/marker.rviz")

    rviz_node = Node(
        condition=IfCondition((use_rviz)),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config_path],
    )
    
    
    video_node = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        output="screen",
    )
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="use_visualization",
                default_value="False",
            ),
            DeclareLaunchArgument(
                name="save_imgs",
                default_value="True",
            ),
            DeclareLaunchArgument(
                name="yolo_model",
                default_value="yolo11n",
            ),
            DeclareLaunchArgument(
                name="use_rviz",
                default_value="True",
                description="Start RViz",
            ),
            sim,
            rviz_node,
            tf_static_node,
            video_node
        ]
    )
