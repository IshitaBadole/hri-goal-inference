import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def generate_launch_description():
    package_dir = get_package_share_directory("my_package")
    robot_description_path = os.path.join(package_dir, "resource", "my_robot.urdf")

    # Declare launch argument for world file
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="my_world.wbt",
        description="World file name to load (should be in worlds/ directory)",
    )

    print(package_dir)

    # Use PathJoinSubstitution to properly join paths with LaunchConfiguration
    world_path = PathJoinSubstitution([
        package_dir,
        "worlds",
        LaunchConfiguration("world")
    ])
    
    webots = WebotsLauncher(world=world_path)

    my_robot_driver = WebotsController(
        robot_name="pedestrian_robot",
        parameters=[
            {"robot_description": robot_description_path},
        ],
    )

    return LaunchDescription(
        [
            world_arg,  # Add the launch argument
            webots,
            my_robot_driver,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessExit(
                    target_action=webots,
                    on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
                )
            ),
        ]
    )
