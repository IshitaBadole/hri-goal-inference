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

            print(f"Moving: Linear={linear_vel:.2f}, Angular={angular_vel:.2f}")
            print(
                f"Position: ({new_pos[0]:.2f}, {new_pos[1]:.2f}), Yaw: {self.__current_yaw:.2f}"
            )
        else:
            if hasattr(self, "_was_moving") and self._was_moving:
                print("Stopped")
                self._was_moving = False

                # Sync physics robot to visual robot position (visual is stable, physics can fall)
                if self.__pedestrian_visual is not None:
                    # Get visual pedestrian position and rotation
                    vis_rotation = self.__pedestrian_visual.getField("rotation")
                    vis_translation = self.__pedestrian_visual.getField("translation")
                    vis_pos = list(vis_translation.getSFVec3f())

                    # Position physics robot slightly lower than visual pedestrian
                    body_pos = [
                        vis_pos[0],
                        vis_pos[1],
                        vis_pos[2] - 0.55,
                    ]  # -0.55m lower

                    # Update physics robot to match visual position
                    body_translation = self.__robot.getSelf().getField("translation")
                    body_rotation = self.__robot.getSelf().getField("rotation")
                    body_translation.setSFVec3f(body_pos)
                    body_rotation.setSFRotation(vis_rotation.getSFRotation())

                    # Update tracked yaw to match
                    vis_rot = vis_rotation.getSFRotation()
                    self.__current_yaw = vis_rot[3]  # Sync tracked angle

        self._was_moving = linear_vel != 0 or angular_vel != 0
