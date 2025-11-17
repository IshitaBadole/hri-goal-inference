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

        # Apply velocity to the robot using physics
        # Get current rotation to determine forward direction
        current_rotation = self.__robot.getSelf().getField("rotation").getSFRotation()
        yaw = current_rotation[3]

        # Calculate velocity components based on orientation
        vel_x = linear_vel * math.cos(yaw)
        vel_y = linear_vel * math.sin(yaw)

        # Apply linear and angular velocity
        self.__robot.getSelf().setVelocity([vel_x, vel_y, 0, 0, 0, angular_vel])

        # Optional: Print current state for debugging
        if linear_vel != 0 or angular_vel != 0:
            position = self.__robot.getSelf().getField("translation").getSFVec3f()
            print(
                f"Target velocity - Linear: {linear_vel:.2f}, Angular: {angular_vel:.2f}"
            )
            print(f"Position: ({position[0]:.2f}, {position[1]:.2f})")
