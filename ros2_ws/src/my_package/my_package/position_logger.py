#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PositionLogger(Node):
    def __init__(self):
        super().__init__("position_logger")

        # Setup log directory
        if os.path.exists("/root/shared"):
            log_dir = "/root/shared/pedestrian_logs"
        else:
            log_dir = os.path.expanduser("~/pedestrian_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Declare the parameters first
        self.declare_parameter('world', 'default_world.wbt')
        self.declare_parameter('participant_id', 'p_default')
        self.declare_parameter('scenario_no', 0)

        #Get the values of the parameters
        self.world_name = self.get_parameter('world').get_parameter_value().string_value
        self.participant_id = self.get_parameter('participant_id').get_parameter_value().string_value
        self.scenario_no = self.get_parameter('scenario_no').get_parameter_value().string_value

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.participant_id}_{self.world_name.replace('.wbt','')}_scenario{self.scenario_no}_{timestamp}.csv"
        self.log_file = os.path.join(log_dir, log_filename)

        # Open with immediate writing
        self.log_handle = open(self.log_file, "w", newline="", buffering=1)
        self.csv_writer = csv.writer(self.log_handle)

        # Write header
        self.csv_writer.writerow(["timestamp", "x", "y", "z", "orientation_z"])
        self.log_handle.flush()

        # Subscribe to pose topic
        self.subscription = self.create_subscription(
            PoseStamped, "/pedestrian_pose", self.pose_callback, 10
        )

        self.get_logger().info(f"Position logger started: {self.log_file}")

    def pose_callback(self, msg):
        """Log received pose data"""
        try:
            timestamp = datetime.now().isoformat()
            pos = msg.pose.position
            orient = msg.pose.orientation

            self.csv_writer.writerow(
                [
                    timestamp,
                    f"{pos.x:.3f}",
                    f"{pos.y:.3f}",
                    f"{pos.z:.3f}",
                    f"{orient.z:.3f}",
                ]
            )

            # Force immediate write
            self.log_handle.flush()
            os.fsync(self.log_handle.fileno())

        except Exception as e:
            self.get_logger().error(f"Logging error: {e}")

    def __del__(self):
        if hasattr(self, "log_handle"):
            self.log_handle.close()


def main(args=None):
    rclpy.init(args=args)
    logger = PositionLogger()

    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info("Shutting down position logger...")
    finally:
        logger.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
