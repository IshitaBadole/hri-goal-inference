import math

import rclpy
from controller import Supervisor
from geometry_msgs.msg import Twist


class PedestrianRobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        # Movement parameters
        self.__linear_speed = 1.0  # m/s max speed
        self.__angular_speed = 1.5  # rad/s max angular speed

        self.__target_twist = Twist()
        self._was_moving = False  # Track movement state for logging

        rclpy.init(args=None)
        self.__node = rclpy.create_node("pedestrian_robot_driver")
        self.__node.create_subscription(Twist, "cmd_vel", self.__cmd_vel_callback, 1)

        print("Pedestrian robot driver initialized")

    def __cmd_vel_callback(self, twist):
        self.__target_twist = twist

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        # Get movement commands
        linear_vel = self.__target_twist.linear.x * self.__linear_speed
        angular_vel = self.__target_twist.angular.z * self.__angular_speed

        # Get time step for position-based movement
        time_step = self.__robot.getBasicTimeStep() / 1000.0  # Convert to seconds

        if linear_vel != 0 or angular_vel != 0:
            # Get current position and rotation
            translation_field = self.__robot.getSelf().getField("translation")
            rotation_field = self.__robot.getSelf().getField("rotation")

            current_pos = translation_field.getSFVec3f()
            current_rot = rotation_field.getSFRotation()

            # Extract current yaw angle (assuming rotation around Z-axis)
            current_yaw = current_rot[3]

            # Update yaw with angular velocity
            new_yaw = current_yaw + angular_vel * time_step

            # Calculate new position based on linear velocity and current orientation
            dx = linear_vel * math.cos(current_yaw) * time_step
            dy = linear_vel * math.sin(current_yaw) * time_step

            new_pos = [
                current_pos[0] + dx,
                current_pos[1] + dy,
                current_pos[2],  # Keep Z constant
            ]

            # Apply new position and rotation
            translation_field.setSFVec3f(new_pos)
            rotation_field.setSFRotation([0, 0, 1, new_yaw])

            print(f"Moving: Linear={linear_vel:.2f}, Angular={angular_vel:.2f}")
            print(f"Position: ({new_pos[0]:.2f}, {new_pos[1]:.2f}), Yaw: {new_yaw:.2f}")
        else:
            # Optional: print when stopped
            if hasattr(self, "_was_moving") and self._was_moving:
                print("Stopped")
                self._was_moving = False

        self._was_moving = linear_vel != 0 or angular_vel != 0
