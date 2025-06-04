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
        executable="rgbd",
        output=["screen"],
        arguments=[
            "/home/dima/diplom_ros/src/orbslam3_ros2/vocabulary/ORBvoc.txt",
            "/home/dima/diplom_ros/src/orbslam3_ros2/config/rgb-d/RealSense_D435i.yaml",
            "false",
        ],
        parameters=[{
            "use_visualization": LaunchConfiguration("use_visualization"),
            "yolo_model": LaunchConfiguration("yolo_model"),
            "save_imgs": LaunchConfiguration("save_imgs"),
            # "DSminPoint": LaunchConfiguration("DSminPoint"),
            # "DSsize": LaunchConfiguration("DSsize")
        }]
    )

    realsensCamera = Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera',
            namespace='camera',
            parameters=[{
                'align_depth': True,
                'enable_color': True,  
                'enable_depth': True,  
            }],
            remappings=[
                ('/camera/realsense2_camera/color/image_raw', 'camera/rgb'),
                ('/camera/realsense2_camera/depth/image_rect_raw', 'camera/depth'),
            ],
            output='screen'  # Вывод логов в терминал
        )


    sim = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "-r 0.5",
            # "--remap",
            # "/cam0/image_raw:=/camera/rgb/image_color",
            "/media/dima/additional/dataset/office6_RGBD",
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

    rviz_node = Node(
        condition=IfCondition((use_rviz)),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config_path],
    )
    run_rviz_node = TimerAction(period=8.0, actions=[rviz_node])
    run_sim_node = TimerAction(period=2.0, actions=[sim])

    return LaunchDescription(
        [
            # DeclareLaunchArgument(
            #     name="DSminPoint",
            #     default_value='4',
            # ),
            # DeclareLaunchArgument(
            #     name="DSsize",
            #     default_value='0.08',
            # ),
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
            orb_slam3_node,
            run_sim_node,
            # realsensCamera,
            tf_static_node,
            run_rviz_node,
        ]
    )
