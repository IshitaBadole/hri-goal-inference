import atexit

# import csv
import math
import os
import signal

import rclpy
from controller import Supervisor
from geometry_msgs.msg import PoseStamped, Twist

# from datetime import datetime


class PedestrianRobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        # Movement parameters
        self.__linear_speed = 0.7  # m/s max speed
        self.__angular_speed = 1.5  # rad/s max angular speed

        self.__target_twist = Twist()

        # Track robot orientation ourselves (don't rely on Webots readback)
        self.__current_yaw = 0.0

        # Get reference to the visual Pedestrian node
        self.__pedestrian_visual = self.__robot.getFromDef("PEDESTRIAN_VIS")
        if self.__pedestrian_visual is None:
            print("Warning: Could not find PEDESTRIAN_VIS node - visual will not sync")
        else:
            print("Found visual pedestrian - will sync position")

        rclpy.init(args=None)
        self.__node = rclpy.create_node("pedestrian_robot_driver")

        # Declare and get parameters
        self.__node.declare_parameter("participant_id", "p_default")
        self.__node.declare_parameter("scenario_no", "0")
        self.__node.declare_parameter("world", "my_world.wbt")

        self.__participant_id = self.__node.get_parameter("participant_id").value
        self.__scenario_no = self.__node.get_parameter("scenario_no").value
        self.__world = self.__node.get_parameter("world").value

        self.__node.create_subscription(Twist, "cmd_vel", self.__cmd_vel_callback, 1)

        # Create publisher for pedestrian pose
        self.__pose_pub = self.__node.create_publisher(
            PoseStamped, "pedestrian_pose", 10
        )

        # Initialize position logging (after node is created to access parameters)
        # self.__log_file_handle = None
        # self.__setup_position_logging()

        print("Pedestrian robot driver initialized")

    # def __setup_position_logging(self):
    #     """Initialize CSV logging for pedestrian positions"""
    #     # Use shared volume if available (Docker), otherwise home directory
    #     if os.path.exists("/root/shared"):
    #         log_dir = "/root/shared/pedestrian_logs"
    #     else:
    #         log_dir = os.path.expanduser("$HOME/pedestrian_logs")

    #     os.makedirs(log_dir, exist_ok=True)

    #     # Create log file with participant_id, world, and scenario_no
    #     # Format: p1_apartment_scenario1.csv
    #     world_name = os.path.splitext(self.__world)[0]  # Remove .wbt extension
    #     self.__log_file = os.path.join(
    #         log_dir,
    #         f"{self.__participant_id}_{world_name}_scenario{self.__scenario_no}.csv",
    #     )

    #     # Write header immediately and close (no persistent handle)
    #     with open(self.__log_file, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(
    #             ["timestamp", "x", "y", "z", "yaw", "linear_vel", "angular_vel"]
    #         )

    #     print(f"Position logging initialized: {self.__log_file}")

    #     # No need for cleanup handlers since we don't keep files open

    def __cmd_vel_callback(self, twist):
        self.__target_twist = twist

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        # Log and publish visual pedestrian position at each step
        self.__publish_position()

        # Get movement commands
        linear_vel = self.__target_twist.linear.x * self.__linear_speed
        angular_vel = self.__target_twist.angular.z * self.__angular_speed

        # Get time step for position-based movement
        time_step = self.__robot.getBasicTimeStep() / 1000.0  # Convert to seconds

        # if robot is moving
        if linear_vel != 0 or angular_vel != 0:
            # Get current position and rotation
            translation_field = self.__robot.getSelf().getField("translation")
            rotation_field = self.__robot.getSelf().getField("rotation")

            current_pos = translation_field.getSFVec3f()

            # Use our tracked yaw instead of reading from Webots (which can cause resets)
            current_yaw = self.__current_yaw

            # Handle pure rotation vs movement with rotation
            new_pos = list(current_pos)  # Start with current position

            # Only calculate linear movement if there's linear velocity
            if linear_vel != 0:
                dx = linear_vel * math.cos(current_yaw) * time_step
                dy = linear_vel * math.sin(current_yaw) * time_step
                new_pos[0] += dx
                new_pos[1] += dy

            # Always maintain proper Z height (prevent falling due to gravity)
            new_pos[2] = 0.72  # Fixed height

            # Update our tracked rotation
            self.__current_yaw += angular_vel * time_step

            # Normalize angle to prevent wrap-around issues
            while self.__current_yaw > math.pi:
                self.__current_yaw -= 2 * math.pi
            while self.__current_yaw < -math.pi:
                self.__current_yaw += 2 * math.pi

            # Apply new position and rotation to physics robot
            translation_field.setSFVec3f(new_pos)
            rotation_field.setSFRotation([0, 0, 1, self.__current_yaw])

            # Sync visual Pedestrian position if available
            if self.__pedestrian_visual is not None:
                vis_translation = self.__pedestrian_visual.getField("translation")
                vis_rotation = self.__pedestrian_visual.getField("rotation")

                # Position visual pedestrian slightly higher than physics robot
                vis_pos = [new_pos[0], new_pos[1], new_pos[2] + 0.55]  # +0.55m higher
                vis_translation.setSFVec3f(vis_pos)
                vis_rotation.setSFRotation([0, 0, 1, self.__current_yaw])

            # print(f"Moving: Linear={linear_vel:.2f}, Angular={angular_vel:.2f}")
            # print(
            #     f"Position: ({new_pos[0]:.2f}, {new_pos[1]:.2f}), Yaw: {self.__current_yaw:.2f}"
            # )
        else:
            # Sync physics robot to visual robot position (visual is stable, physics can fall)
            if self.__pedestrian_visual is not None:
                # Get visual pedestrian position and rotation
                vis_rotation = self.__pedestrian_visual.getField("rotation")
                vis_translation = self.__pedestrian_visual.getField("translation")
                vis_pos = list(vis_translation.getSFVec3f())

                # print(
                #     f"Vis Position: ({vis_pos[0]:.2f}, {vis_pos[1]:.2f}, {vis_pos[2]:.2f})"
                # )
                # print(
                #     f"New Body Position: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})"
                # )

                body_rotation = self.__robot.getSelf().getField("rotation")
                body_rotation_val = list(body_rotation.getSFRotation())
                # print(
                #     f"Body Rotation: ({body_rotation_val[0]:.2f}, {body_rotation_val[1]:.2f}, {body_rotation_val[2]:.2f})"
                # )
                # body has fallen, make it upright and align position to vis
                if body_rotation_val[1] < 0:
                    vis_rot = vis_rotation.getSFRotation()
                    body_rotation.setSFRotation(vis_rot)
                    body_translation = self.__robot.getSelf().getField("translation")
                    # Position physics robot slightly lower than visual pedestrian
                    body_pos = [
                        vis_pos[0],
                        vis_pos[1],
                        vis_pos[2] - 0.55,
                    ]  # -0.55m lower
                    body_translation.setSFVec3f(body_pos)

                    # Update tracked yaw to match
                    self.__current_yaw = vis_rot[3]  # Sync tracked angle

    def __publish_position(self):
        """Publish current visual pedestrian position to ROS topic"""
        if self.__pedestrian_visual is not None:
            try:
                # Get visual pedestrian position and rotation
                vis_translation = self.__pedestrian_visual.getField("translation")
                vis_rotation = self.__pedestrian_visual.getField("rotation")

                if vis_translation is not None and vis_rotation is not None:
                    pos = vis_translation.getSFVec3f()
                    rot = vis_rotation.getSFRotation()

                    # Publish pose message
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.__node.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position.x = pos[0]
                    pose_msg.pose.position.y = pos[1]
                    pose_msg.pose.position.z = pos[2]

                    # Convert yaw to quaternion (z-axis rotation only)
                    cy = math.cos(rot[3] * 0.5)
                    sy = math.sin(rot[3] * 0.5)
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = sy
                    pose_msg.pose.orientation.w = cy

                    self.__pose_pub.publish(pose_msg)

            except Exception as e:
                # Don't let publishing errors crash the simulation
                print(f"Publishing error: {e}")
