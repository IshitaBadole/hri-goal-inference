import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class SimpleTeleop(Node):
    def __init__(self):
        super().__init__("pedestrian_teleop")
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.get_logger().info(
            "Pedestrian teleop started. Use WASD, Q/E for diagonal, X to stop, CTRL+C to quit."
        )

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        twist = Twist()
        linear_speed = 0.8  # Increased for pedestrian movement
        angular_speed = 1.2  # Increased for more responsive turning

        print("Pedestrian teleop controls:")
        print("  W/S: Move forward/backward")
        print("  A/D: Turn left/right")
        print("  Q/E: Move diagonally")
        print("  X: Stop")
        print("  CTRL+C: Quit")

        while rclpy.ok():
            key = self.get_key()

            if key == "w":
                twist.linear.x = linear_speed
                twist.angular.z = 0.0
            elif key == "s":
                twist.linear.x = -linear_speed
                twist.angular.z = 0.0
            elif key == "a":
                twist.linear.x = 0.0
                twist.angular.z = angular_speed
            elif key == "d":
                twist.linear.x = 0.0
                twist.angular.z = -angular_speed
            elif key == "q":
                twist.linear.x = linear_speed
                twist.angular.z = angular_speed
            elif key == "e":
                twist.linear.x = linear_speed
                twist.angular.z = -angular_speed
            elif key == "x":
                twist = Twist()  # stop
            elif key == "\x03":  # CTRL+C
                break

            self.pub.publish(twist)


def main():
    rclpy.init()
    node = SimpleTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
