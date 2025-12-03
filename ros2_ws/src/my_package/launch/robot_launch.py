import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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

    # Log file name constructed using world, participant_id, and scenario_no (Ex : p1_apartment_scenario1.log)

    # Declare launch argument for participant ID
    participant_arg = DeclareLaunchArgument(
        "participant_id",
        default_value="p_default",
        description="Participant ID for the simulation (used to name log file)",
    )

    # Declare launch argument for Scenario number
    scenario_arg = DeclareLaunchArgument(
        "scenario_no",
        default_value="0",
        description="Scenario number being simulated in the current world (used to name log file)",
    )

    # Get the launch configurations
    world = LaunchConfiguration("world")
    participant_id = LaunchConfiguration("participant_id")
    scenario_no = LaunchConfiguration("scenario_no")

    print(f"World file: {world}")
    print(f"Participant ID: {participant_id}")
    print(f"Scenario number: {scenario_no}")

    print(package_dir)

    # Use PathJoinSubstitution to properly join paths with LaunchConfiguration
    world_path = PathJoinSubstitution([package_dir, "worlds", world])

    webots = WebotsLauncher(world=world_path)

    my_robot_driver = WebotsController(
        robot_name="pedestrian_robot",
        parameters=[
            {"robot_description": robot_description_path},
            {"participant_id": participant_id},
            {"scenario_no": scenario_no},
            {"world": world},
        ],
    )

    position_logger_node = Node(
        package="my_package",
        executable="position_logger",
        name="position_logger",
        output="screen",
        parameters=[
            {"world": world},
            {"participant_id": participant_id},
            {"scenario_no": scenario_no},
        ],
    )

    return LaunchDescription(
        [
            world_arg,  # Add the launch arguments
            participant_arg,
            scenario_arg,
            position_logger_node,
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
