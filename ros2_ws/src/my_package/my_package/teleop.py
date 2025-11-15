import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class SimpleTeleop(Node):
    def __init__(self):
        super().__init__('simple_teleop')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("Simple teleop started. Use WASD, Q/E for diagonal, X to stop, CTRL+C to quit.")

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
        linear_speed = 0.3
        angular_speed = 1.0

        while rclpy.ok():
            key = self.get_key()

            if key == 'w':
                twist.linear.x = linear_speed
                twist.angular.z = 0.0
            elif key == 's':
                twist.linear.x = -linear_speed
                twist.angular.z = 0.0
            elif key == 'a':
                twist.linear.x = 0.0
                twist.angular.z = angular_speed
            elif key == 'd':
                twist.linear.x = 0.0
                twist.angular.z = -angular_speed
            elif key == 'q':
                twist.linear.x = linear_speed
                twist.angular.z = angular_speed
            elif key == 'e':
                twist.linear.x = linear_speed
                twist.angular.z = -angular_speed
            elif key == 'x':
                twist = Twist()  # stop
            elif key == '\x03':  # CTRL+C
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


if __name__ == '__main__':
    main()