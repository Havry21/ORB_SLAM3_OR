from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    sim = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "--loop",
            "/media/dima/additional/dataset/kitchen",
        ],
        output="screen",
    )
    
    image_show = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
    )

    return LaunchDescription(
        [
            sim,
            image_show
        ]
    )
