# controllers/ped_body/ped_body.py
import math
from collections import deque

from controller import Supervisor

BODY_Z = 0.72
VIS_Z = 1.27
STEP_M = 0.02  # Reduced from 0.04 to 0.02 for more realistic pedestrian speed
RADIUS_M = 0.25
STOP_EPS = 0.3  # Increased goal threshold - stop when within 30cm of goal
MIN_BLOCKED_DISTANCE = (
    0.6  # Increased from 0.4m - minimum distance to maintain from blocked goals
)
MAX_BLOCKED_DISTANCE = (
    1.2  # Increased from 1.0m - maximum distance to search for closest approach
)
RAY_EPS = 0.01
WALL_BUFFER = 0.08  # Reduced safety buffer around walls (8cm for testing)
GOAL_WAIT_TIME = 5.0  # Wait 5 seconds at each goal

# BFS pathfinding constants
BFS_GRID_SIZE = 0.2  # 20cm grid cells for BFS pathfinding
BFS_SEARCH_RADIUS = 2.0  # Search within 2m radius for alternative paths
BFS_BACKUP_DISTANCE = (
    0.4  # Distance to backup when stuck (reduced to 40cm for less aggressive backing)
)


class PedBody(Supervisor):
    def __init__(self):
        super().__init__()
        self.dt = int(self.getBasicTimeStep())
        self.body = self.getSelf()
        self.tr = self.body.getField("translation")
        self.rot = self.body.getField("rotation")

        # optional visual
        self.vis = self.getFromDef("PED_VIS")
        self.vis_tr = self.vis.getField("translation") if self.vis else None
        self.vis_rot = self.vis.getField("rotation") if self.vis else None

        # Waypoints
        # apartment goals
        # self.goals = [(-4.5, -6.8), (0, -1.65), (-8.25, -6.6)]
        # self.goals = [(-4.5, -6.8), (-0.7, -4.9), (0, -1.6), (-8.25, -6.6)]
        # factory goals
        # self.goals = [(8.1, 1.8), (0.95, -10), (2, 3.25)]
        # hall goals: big valve on cabinet, telephone(2), robot(1) emergency button, first aid box
        # self.goals = [(-0.28, 10.3), (3.23, -1.006), (-0.25, -10.0), (-3.2, -9.92)]
        # break room: sofa, sink, door
        self.goals = [(-2.2, -1.5), (0.8, -0.2), (0, 4)]
        self.gi = 0

        # Add velocity control for physics-based movement
        self.max_velocity = 0.5  # m/s
        self.use_physics_movement = (
            False  # Changed to False - use direct position control
        )

        # Obstacle avoidance parameters
        self.avoidance_timer = 0  # How long to maintain avoidance behavior
        self.max_avoidance_time = 40  # Reduced from 60 to prevent long circular motion
        self.wall_follow_side = 1  # 1 for right, -1 for left wall following

        # Simple stuck detection
        self.stuck_counter = 0  # Track if robot is stuck
        self.last_position = [0, 0]  # Previous position for stuck detection
        self.stuck_threshold = 10  # Reduced threshold - detect stuck faster
        self.backup_timer = 0  # How long to walk backwards
        self.corner_escape_timer = 0  # Special timer for corner situations
        self.collision_detection_enabled = (
            True  # Will be set to False if no raycast available
        )

        # Goal waiting parameters
        self.at_goal = False  # Whether we've reached the current goal
        self.goal_wait_timer = 0  # How long we've been waiting at goal
        self.goal_wait_steps = int(
            GOAL_WAIT_TIME * 1000 / self.dt
        )  # Convert seconds to steps
        self.goal_wait_position = (
            None  # Store position when goal is reached to prevent drift
        )
        self.goal_wait_angle = (
            0.0  # Store orientation when goal is reached to face the goal
        )

        # Debug: Print the calculated wait steps
        print(
            f"Goal wait configuration: {GOAL_WAIT_TIME}s = {self.goal_wait_steps} steps (dt={self.dt}ms)"
        )  # BFS pathfinding parameters
        self.bfs_path = []  # Current BFS path to follow
        self.bfs_target_index = 0  # Current target in BFS path
        self.bfs_stuck_counter = 0  # Counter for when BFS path fails
        self.last_bfs_position = [0, 0]  # For detecting if BFS movement is stuck
        self.exploration_mode = False  # Whether we're in BFS exploration mode

        # Mission completion parameters
        self.all_goals_completed = False  # Track if all goals have been reached
        self.mission_start_gi = 0  # Track the starting goal index

        # Goal-oriented movement parameters
        self.goal_oriented_mode = True  # Enable goal-oriented movement behavior
        self.current_orientation_target = None  # Target angle we're trying to face
        self.orientation_tolerance = (
            math.pi / 12
        )  # 15 degrees tolerance for facing goal
        self.is_facing_goal = False  # Whether robot is currently facing the goal
        self.reorient_after_avoidance = (
            False  # Whether to reorient after obstacle avoidance
        )

        # ---- Raycast function: may be on Supervisor or on Node ----
        self._supervisor_ray = getattr(self, "rayCast", None)  # Supervisor.rayCast?

        self._root = self.getRoot()
        self._node_ray = getattr(self._root, "rayCast", None) or getattr(
            self.body, "rayCast", None
        )

        # Try additional raycast sources
        if not self._supervisor_ray and not self._node_ray:
            # Try getting raycast from the robot node directly
            try:
                self._node_ray = self.body.rayCast
            except AttributeError:
                pass

            # Try getting it from the supervisor
            if not self._node_ray:
                try:
                    self._supervisor_ray = self.rayCast
                except AttributeError:
                    pass

        # Debug raycast availability
        if self._supervisor_ray:
            print("Using Supervisor raycast")
        elif self._node_ray:
            print("Using Node raycast")
        else:
            print("WARNING: No raycast available - using simple collision detection")
            # For now, disable collision detection when raycast is not available
            self.collision_detection_enabled = False
            print(
                "Pedestrian will move without collision detection - may pass through walls"
            )

    def _rc(self, p_from, p_to):
        """Unified raycast: returns dict with 'point','normal','node' or None."""
        hit = None
        if self._supervisor_ray:
            hit = self._supervisor_ray(p_from, p_to)
        elif self._node_ray:
            hit = self._node_ray(p_from, p_to)
        if hit and "point" in hit:
            return hit
        return None

    def _check_collision(self, from_pos, to_pos):
        """Check for collision along a path at multiple heights with wall buffer"""
        if not self.collision_detection_enabled:
            return False  # No collision detection available

        if not (self._supervisor_ray or self._node_ray):
            return False

        heights_to_check = [0.1, 0.5, BODY_Z, 1.0, 1.5]
        for check_height in heights_to_check:
            hit = self._rc(
                [from_pos[0], from_pos[1], check_height],
                [to_pos[0], to_pos[1], check_height],
            )
            if hit:
                # Calculate distance to collision point
                hit_point = hit["point"]
                distance_to_hit = math.hypot(
                    hit_point[0] - from_pos[0], hit_point[1] - from_pos[1]
                )

                # Apply buffer - consider collision if within buffer distance
                total_buffer = RADIUS_M + WALL_BUFFER
                if distance_to_hit <= total_buffer:
                    return True

        return False

    def _find_closest_safe_approach(self, current_x, current_y, goal_x, goal_y):
        """Find the closest safe distance we can get to a blocked goal"""
        if not self.collision_detection_enabled:
            return STOP_EPS  # Default to normal threshold if no collision detection

        # Calculate direction to goal
        dx = goal_x - current_x
        dy = goal_y - current_y
        goal_distance = math.hypot(dx, dy)

        if goal_distance < 1e-9:
            return STOP_EPS

        # Normalize direction
        nx, ny = dx / goal_distance, dy / goal_distance

        # Binary search for the closest safe distance
        min_safe_distance = MIN_BLOCKED_DISTANCE
        max_test_distance = min(goal_distance, MAX_BLOCKED_DISTANCE)

        # If we're already closer than minimum safe distance, use current distance
        if goal_distance <= min_safe_distance:
            return goal_distance * 0.95  # Slightly less than current distance

        # Binary search to find closest approach
        safe_distance = max_test_distance
        step_size = 0.05  # 5cm increments

        for test_distance in [
            d * step_size
            for d in range(
                int(min_safe_distance / step_size),
                int(max_test_distance / step_size) + 1,
            )
        ]:
            # Calculate test position
            test_x = current_x + nx * (goal_distance - test_distance)
            test_y = current_y + ny * (goal_distance - test_distance)

            # Check if we can safely reach this position
            if not self._check_collision_with_buffer(
                [current_x, current_y], [test_x, test_y]
            ):
                safe_distance = test_distance
                break

        return safe_distance

    def _world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(round(x / BFS_GRID_SIZE))
        grid_y = int(round(y / BFS_GRID_SIZE))
        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * BFS_GRID_SIZE
        world_y = grid_y * BFS_GRID_SIZE
        return world_x, world_y

    def _is_grid_cell_free(self, grid_x, grid_y):
        """Check if a grid cell is free of obstacles"""
        world_x, world_y = self._grid_to_world(grid_x, grid_y)

        # Check multiple points within the grid cell for thorough collision detection
        test_points = [
            (world_x, world_y),  # Center
            (world_x - BFS_GRID_SIZE / 3, world_y - BFS_GRID_SIZE / 3),  # Corners
            (world_x + BFS_GRID_SIZE / 3, world_y - BFS_GRID_SIZE / 3),
            (world_x - BFS_GRID_SIZE / 3, world_y + BFS_GRID_SIZE / 3),
            (world_x + BFS_GRID_SIZE / 3, world_y + BFS_GRID_SIZE / 3),
        ]

        for test_x, test_y in test_points:
            # Check if this point has a collision
            if self._check_collision_with_buffer(
                [world_x, world_y], [test_x, test_y], extra_buffer=0.05
            ):
                return False

        return True

    def _bfs_pathfind(self, start_x, start_y, goal_x, goal_y):
        """Find path using BFS from start to goal"""
        if not self.collision_detection_enabled:
            return []  # No pathfinding without collision detection

        start_grid = self._world_to_grid(start_x, start_y)
        goal_grid = self._world_to_grid(goal_x, goal_y)

        # BFS setup
        queue = deque([(start_grid, [])])
        visited = {start_grid}

        # Directions: North, East, South, West, and diagonals
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),  # Cardinal directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),  # Diagonal directions
        ]

        max_iterations = 100  # Limit search to prevent infinite loops
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            current_pos, path = queue.popleft()

            # Check if we've reached the goal (within reasonable distance)
            goal_distance = math.hypot(
                current_pos[0] - goal_grid[0], current_pos[1] - goal_grid[1]
            )

            if goal_distance <= 2:  # Within 2 grid cells of goal
                # Convert path back to world coordinates
                world_path = []
                for grid_pos in path + [current_pos]:
                    world_x, world_y = self._grid_to_world(grid_pos[0], grid_pos[1])
                    world_path.append((world_x, world_y))
                return world_path

            # Explore neighbors
            for dx, dy in directions:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if new_pos not in visited:
                    # Check if this grid cell is free
                    if self._is_grid_cell_free(new_pos[0], new_pos[1]):
                        visited.add(new_pos)
                        new_path = path + [current_pos]
                        queue.append((new_pos, new_path))

        return []  # No path found

    def _is_facing_goal(self, current_x, current_y, goal_x, goal_y):
        """Check if robot is currently facing toward the goal within tolerance"""
        # Calculate desired angle to goal
        dx, dy = goal_x - current_x, goal_y - current_y
        desired_angle = math.atan2(dy, dx)

        # Get current robot orientation (from rotation field)
        current_rotation = self.rot.getSFRotation()
        current_angle = current_rotation[3]  # Rotation angle around Z-axis

        # Calculate angle difference
        angle_diff = abs(desired_angle - current_angle)
        # Normalize to [0, pi]
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        return angle_diff <= self.orientation_tolerance

    def _orient_toward_goal(self, current_x, current_y, goal_x, goal_y):
        """Orient robot to face the goal direction"""
        dx, dy = goal_x - current_x, goal_y - current_y
        goal_angle = math.atan2(dy, dx)

        # Set robot orientation toward goal
        self._set_pose(current_x, current_y, goal_angle)
        self.current_orientation_target = goal_angle

        return goal_angle

    def _execute_bfs_navigation(self, current_x, current_y, goal_x, goal_y):
        """Execute BFS-based navigation when stuck"""
        # Check if we need to compute a new path
        if not self.bfs_path or self.bfs_target_index >= len(self.bfs_path):
            print("Computing BFS path to goal...")
            self.bfs_path = self._bfs_pathfind(current_x, current_y, goal_x, goal_y)
            self.bfs_target_index = 0

            if not self.bfs_path:
                print("No BFS path found, falling back to backup behavior")
                return None

            print(f"Found BFS path with {len(self.bfs_path)} waypoints")

        # Check if we're stuck on current BFS path
        bfs_distance_moved = math.hypot(
            current_x - self.last_bfs_position[0], current_y - self.last_bfs_position[1]
        )

        if bfs_distance_moved < STEP_M * 0.3:
            self.bfs_stuck_counter += 1
        else:
            self.bfs_stuck_counter = 0

        self.last_bfs_position = [current_x, current_y]

        # If stuck on BFS path for too long, recompute
        if self.bfs_stuck_counter > 20:
            print("BFS path seems blocked, recomputing...")
            self.bfs_path = []
            self.bfs_stuck_counter = 0
            return None

        # Get current target from BFS path
        if self.bfs_target_index < len(self.bfs_path):
            target_x, target_y = self.bfs_path[self.bfs_target_index]

            # Check if we've reached current target
            target_distance = math.hypot(target_x - current_x, target_y - current_y)

            if target_distance <= BFS_GRID_SIZE * 0.8:  # Reached current waypoint
                self.bfs_target_index += 1
                print(
                    f"Reached BFS waypoint {self.bfs_target_index}/{len(self.bfs_path)}"
                )

                # Get next target if available
                if self.bfs_target_index < len(self.bfs_path):
                    target_x, target_y = self.bfs_path[self.bfs_target_index]
                else:
                    # Reached end of path, switch back to direct navigation
                    print("Completed BFS path, switching to direct navigation")
                    self.exploration_mode = False
                    self.bfs_path = []
                    return None

            # Calculate direction to current BFS target
            dx, dy = target_x - current_x, target_y - current_y
            return math.atan2(dy, dx)

        return None

    def _is_goal_blocked(self, current_x, current_y, goal_x, goal_y):
        """Check if the direct path to the goal is blocked by obstacles and return safe distance"""
        if not self.collision_detection_enabled:
            return (
                False,
                STOP_EPS,
            )  # Can't determine if blocked without collision detection

        # Check if there's a collision along the direct path to the goal
        is_blocked = self._check_collision_with_buffer(
            [current_x, current_y], [goal_x, goal_y], extra_buffer=0.0
        )

        if is_blocked:
            # Find the closest safe approach distance
            safe_distance = self._find_closest_safe_approach(
                current_x, current_y, goal_x, goal_y
            )
            return True, safe_distance
        else:
            return False, STOP_EPS

    def _check_collision_with_buffer(self, from_pos, to_pos, extra_buffer=0.0):
        """Check collision with additional custom buffer"""
        if not self.collision_detection_enabled:
            return False  # No collision detection available

        if not (self._supervisor_ray or self._node_ray):
            return False

        heights_to_check = [0.1, 0.5, BODY_Z, 1.0, 1.5]
        for check_height in heights_to_check:
            hit = self._rc(
                [from_pos[0], from_pos[1], check_height],
                [to_pos[0], to_pos[1], check_height],
            )
            if hit:
                hit_point = hit["point"]
                distance_to_hit = math.hypot(
                    hit_point[0] - from_pos[0], hit_point[1] - from_pos[1]
                )

                # Apply buffer plus extra buffer
                total_buffer = RADIUS_M + WALL_BUFFER + extra_buffer

                if distance_to_hit <= total_buffer:
                    return True

        return False

    def _is_stuck(self, current_x, current_y):
        """Detect if robot is stuck by checking movement over time and circular motion"""
        distance_moved = math.hypot(
            current_x - self.last_position[0], current_y - self.last_position[1]
        )

        # Track position history for circular motion detection
        if not hasattr(self, "position_history"):
            self.position_history = []

        self.position_history.append((current_x, current_y))
        if len(self.position_history) > 15:  # Keep last 15 positions (reduced from 20)
            self.position_history.pop(0)

        # Check for circular motion - if robot returns near a previous position
        circular_motion_detected = False
        if len(self.position_history) >= 8:  # Reduced from 15 to 8 for faster detection
            recent_pos = (current_x, current_y)
            for i, old_pos in enumerate(
                self.position_history[
                    :-5
                ]  # Check against positions from 5+ steps ago (reduced from 10)
            ):
                old_distance = math.hypot(
                    recent_pos[0] - old_pos[0], recent_pos[1] - old_pos[1]
                )
                if (
                    old_distance < STEP_M * 4
                ):  # Within 8cm of old position (increased from 6cm)
                    circular_motion_detected = True
                    print(
                        f"Circular motion detected: returned to position from {len(self.position_history) - i} steps ago"
                    )
                    break

        if (
            distance_moved < STEP_M * 0.2 or circular_motion_detected
        ):  # Very little movement OR circular motion
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_position = [current_x, current_y]
        return self.stuck_counter > self.stuck_threshold

    def _is_in_corner(self, x, y):
        """Detect if robot is in a corner by checking how many directions are blocked"""
        check_distance = STEP_M * 3  # Look further ahead
        directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]  # N, E, S, W
        blocked_count = 0

        for angle in directions:
            test_x = x + math.cos(angle) * check_distance
            test_y = y + math.sin(angle) * check_distance
            # Use buffered collision detection for corner detection
            if self._check_collision_with_buffer(
                [x, y], [test_x, test_y], extra_buffer=0.05
            ):
                blocked_count += 1

        return blocked_count >= 2  # Reduced threshold - corner if 2+ directions blocked

    def _get_backup_direction(self, target_x, target_y, current_x, current_y):
        """Get direction opposite to the goal for backing up, but check if it's clear"""
        dx, dy = target_x - current_x, target_y - current_y
        backup_angle = math.atan2(dy, dx) + math.pi  # Opposite direction

        # Check if backup direction is clear (with buffer)
        backup_distance = STEP_M * 3
        backup_x = current_x + math.cos(backup_angle) * backup_distance
        backup_y = current_y + math.sin(backup_angle) * backup_distance

        if not self._check_collision_with_buffer(
            [current_x, current_y], [backup_x, backup_y]
        ):
            return backup_angle

        # If backing up is blocked, try side-stepping
        # Try perpendicular directions (left and right relative to goal direction)
        goal_angle = math.atan2(dy, dx)
        side_angles = [
            goal_angle + math.pi / 2,  # Left
            goal_angle - math.pi / 2,  # Right
            goal_angle + 3 * math.pi / 4,  # Back-left
            goal_angle - 3 * math.pi / 4,  # Back-right
        ]

        for angle in side_angles:
            side_x = current_x + math.cos(angle) * backup_distance
            side_y = current_y + math.sin(angle) * backup_distance
            if not self._check_collision_with_buffer(
                [current_x, current_y], [side_x, side_y]
            ):
                return angle

        # If all directions blocked, return original backup direction (emergency)
        return backup_angle

    def _find_avoidance_direction(self, x, y, target_x, target_y):
        """Find a direction to avoid obstacles using wall-following behavior with buffer"""
        # Calculate desired direction
        dx, dy = target_x - x, target_y - y
        target_angle = math.atan2(dy, dx)

        # Check different angles around the robot
        check_angles = [
            target_angle + math.pi / 4 * self.wall_follow_side,  # 45 degrees
            target_angle + math.pi / 2 * self.wall_follow_side,  # 90 degrees
            target_angle + 3 * math.pi / 4 * self.wall_follow_side,  # 135 degrees
            target_angle + math.pi * self.wall_follow_side,  # 180 degrees
            target_angle - math.pi / 4 * self.wall_follow_side,  # -45 degrees
        ]

        check_distance = STEP_M * 4  # Look further ahead

        for angle in check_angles:
            test_x = x + math.cos(angle) * check_distance
            test_y = y + math.sin(angle) * check_distance

            # Check if this direction is clear (with buffer)
            if not self._check_collision_with_buffer([x, y], [test_x, test_y]):
                # Also check if we can take a few steps in this direction
                clear_path = True
                for i in range(1, 5):  # Check next 5 steps
                    step_x = x + math.cos(angle) * STEP_M * i
                    step_y = y + math.sin(angle) * STEP_M * i
                    if self._check_collision_with_buffer([x, y], [step_x, step_y]):
                        clear_path = False
                        break

                if clear_path:
                    return angle

        # If no clear direction found, try a few more angles before switching sides
        # Try diagonal angles with current side
        fallback_angles = [
            target_angle + math.pi / 6 * self.wall_follow_side,  # 30 degrees
            target_angle + 2 * math.pi / 3 * self.wall_follow_side,  # 120 degrees
            target_angle + 5 * math.pi / 6 * self.wall_follow_side,  # 150 degrees
        ]

        for angle in fallback_angles:
            test_x = x + math.cos(angle) * check_distance
            test_y = y + math.sin(angle) * check_distance
            if not self._check_collision_with_buffer([x, y], [test_x, test_y]):
                return angle

        # Only switch sides as last resort, but don't do it too often
        if not hasattr(self, "last_side_switch_time"):
            self.last_side_switch_time = 0

        # Limit side switching to once every 2 seconds (roughly 125 simulation steps)
        if hasattr(self, "simulation_step_counter"):
            self.simulation_step_counter += 1
        else:
            self.simulation_step_counter = 0

        if self.simulation_step_counter - self.last_side_switch_time > 125:
            self.wall_follow_side *= -1
            self.last_side_switch_time = self.simulation_step_counter

        return target_angle + math.pi / 2 * self.wall_follow_side

    def _set_pose(self, x, y, yaw):
        if self.use_physics_movement:
            # Use velocity-based movement (respects physics collisions)
            current_pos = self.tr.getSFVec3f()
            dx = x - current_pos[0]
            dy = y - current_pos[1]

            # Set linear velocity
            velocity = [dx * 10, dy * 10, 0]  # Scale factor for responsive movement
            self.body.setVelocity(velocity)

            # Set rotation
            self.rot.setSFRotation([0, 0, 1, yaw])
        else:
            # Direct position control (bypasses physics)
            self.tr.setSFVec3f([x, y, BODY_Z])
            self.rot.setSFRotation([0, 0, 1, yaw])

        # Always update visual representation
        if self.vis_tr:
            self.vis_tr.setSFVec3f([x, y, VIS_Z])
        if self.vis_rot:
            self.vis_rot.setSFRotation([0, 0, 1, yaw])

    def _advance_toward(self, x, y, tx, ty):
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return x, y, 0.0, True
        nx, ny = dx / dist, dy / dist
        step = min(STEP_M, dist)
        nxp, nyp = x + nx * step, y + ny * step

        # Use the buffered collision detection system
        if self._check_collision_with_buffer([x, y], [nxp, nyp]):
            print(f"Collision detected, stopping at ({x:.2f}, {y:.2f})")
            yaw = math.atan2(ny, nx)
            return x, y, yaw, True  # Stop if collision detected

        yaw = math.atan2(ny, nx)
        arrived = (dist - step) <= STOP_EPS
        print(f"Moving from ({x:.2f}, {y:.2f}) to ({nxp:.2f}, {nyp:.2f})")
        return nxp, nyp, yaw, arrived

    def run(self):
        x, y, _ = self.tr.getSFVec3f()
        yaw = 0.0
        self._set_pose(x, y, yaw)

        while self.step(self.dt) != -1:
            # Check if mission is complete
            if self.all_goals_completed:
                # Print completion message occasionally (every 5 seconds)
                if not hasattr(self, "last_completion_message_time"):
                    self.last_completion_message_time = 0

                self.last_completion_message_time += self.dt
                if self.last_completion_message_time >= 5000:  # 5 seconds in ms
                    print(
                        "✅ All goals completed. Robot is stationed at final position."
                    )
                    self.last_completion_message_time = 0

                # Ensure robot stays stopped and maintains position
                if self.use_physics_movement:
                    self.body.setVelocity([0, 0, 0])

                # Lock robot position at completion to prevent falling
                if hasattr(self, "completion_position") and hasattr(
                    self, "completion_angle"
                ):
                    self._set_pose(
                        self.completion_position[0],
                        self.completion_position[1],
                        self.completion_angle,
                    )
                else:
                    # Store completion position if not already stored
                    current_pos = self.tr.getSFVec3f()
                    current_rot = self.rot.getSFRotation()
                    self.completion_position = [current_pos[0], current_pos[1]]
                    self.completion_angle = current_rot[3]  # Z-axis rotation angle

                continue  # Stay in loop but don't move

            # Get current position (important for physics-based movement)
            current_pos = self.tr.getSFVec3f()
            x, y = current_pos[0], current_pos[1]

            tx, ty = self.goals[self.gi]

            # Print current goal (only when it changes)
            if (
                not hasattr(self, "last_printed_goal")
                or self.last_printed_goal != self.gi
            ):
                print(f"Moving to Goal {self.gi}: ({tx:.2f}, {ty:.2f})")
                self.last_printed_goal = self.gi

            # Add basic movement debug
            dist = math.hypot(tx - x, ty - y)

            # Check if the direct path to goal is blocked and get appropriate stopping distance
            goal_blocked, effective_stop_eps = self._is_goal_blocked(x, y, tx, ty)

            # Debug output for blocked goals
            if goal_blocked and not hasattr(self, "goal_blocked_printed"):
                print(
                    f"Goal {self.gi} is blocked (e.g., apple on table). Will stop at {effective_stop_eps:.2f}m distance."
                )
                self.goal_blocked_printed = True
            elif not goal_blocked and hasattr(self, "goal_blocked_printed"):
                delattr(self, "goal_blocked_printed")

            # Check if we've reached the goal (within appropriate threshold)
            if dist <= effective_stop_eps:
                if not self.at_goal:
                    # Just reached the goal
                    self.at_goal = True
                    self.goal_wait_timer = 0
                    # Lock the robot's position to prevent drift
                    self.goal_wait_position = [x, y]
                    # Calculate angle to face the goal
                    if goal_blocked:
                        # For blocked goals, face toward the actual goal coordinates
                        dx, dy = tx - x, ty - y
                        self.goal_wait_angle = math.atan2(dy, dx)
                        print(
                            f"Reached closest approach to Goal {self.gi} at ({x:.2f}, {y:.2f}) - {dist:.2f}m away (goal blocked), waiting {GOAL_WAIT_TIME} seconds..."
                        )
                    else:
                        # For accessible goals, face toward goal or maintain current heading
                        dx, dy = tx - x, ty - y
                        if (
                            math.hypot(dx, dy) > 0.01
                        ):  # If there's a meaningful direction
                            self.goal_wait_angle = math.atan2(dy, dx)
                        else:
                            # If very close, maintain current heading or face forward
                            self.goal_wait_angle = (
                                0.0  # Face forward (positive X direction)
                            )
                        print(
                            f"Reached Goal {self.gi} at ({tx:.2f}, {ty:.2f}), waiting {GOAL_WAIT_TIME} seconds..."
                        )
                    # Stop all movement immediately and lock position
                    if self.use_physics_movement:
                        self.body.setVelocity([0, 0, 0])
                    # Force set position and orientation to face the goal
                    self._set_pose(
                        self.goal_wait_position[0],
                        self.goal_wait_position[1],
                        self.goal_wait_angle,
                    )

                # Wait at goal - increment timer and check if waiting is complete
                self.goal_wait_timer += 1
                if self.goal_wait_timer >= self.goal_wait_steps:
                    # Finished waiting, move to next goal
                    old_gi = self.gi
                    next_gi = (self.gi + 1) % len(self.goals)

                    # Check if we've completed all goals (returned to starting goal)
                    if (
                        next_gi == self.mission_start_gi
                        and self.gi != self.mission_start_gi
                    ):
                        # All goals completed!
                        self.all_goals_completed = True
                        print(
                            f"🎉 Mission Complete! All {len(self.goals)} goals have been reached."
                        )
                        print("Robot stopping at final goal location.")
                        # Stop all movement but continue simulation loop
                        if self.use_physics_movement:
                            self.body.setVelocity([0, 0, 0])
                        # Don't return - let the mission completion logic at start of loop handle it
                        continue
                    else:
                        # Continue to next goal
                        new_goal_x, new_goal_y = self.goals[next_gi]
                        print(
                            f"Goal {old_gi} completed! Moving to goal {next_gi} at ({new_goal_x:.2f}, {new_goal_y:.2f})"
                        )
                        print(
                            f"Robot current position when transitioning: ({x:.2f}, {y:.2f})"
                        )

                        self.gi = next_gi
                        self.at_goal = False
                        self.goal_wait_timer = 0
                        self.goal_wait_position = None  # Reset locked position
                        self.goal_wait_angle = 0.0  # Reset locked orientation
                        # Reset all avoidance states
                        self.avoidance_timer = 0
                        self.backup_timer = 0
                        self.corner_escape_timer = 0
                        self.stuck_counter = 0
                        # Reset BFS states
                        self.exploration_mode = False
                        self.bfs_path = []
                        self.bfs_target_index = 0
                        self.bfs_stuck_counter = 0
                        # Reset goal-oriented movement flags for new goal
                        self.is_facing_goal = False
                        self.reorient_after_avoidance = False
                        self.current_orientation_target = None

                        # Check if robot is in a problematic position (too close to obstacles) after goal completion
                        # This helps if the robot got stuck near an object at the previous goal
                        if self._is_in_corner(x, y):
                            print(
                                "Robot seems trapped after goal completion, initiating escape sequence"
                            )
                            self.backup_timer = 8  # Reduced backup movement (16cm)
                            self.reorient_after_avoidance = True
                        print(f"Goal {old_gi} completed! Moving to goal {self.gi}")
                else:
                    # Still waiting at goal, don't move
                    remaining_time = (
                        (self.goal_wait_steps - self.goal_wait_timer) * self.dt / 1000.0
                    )
                    # More frequent debug output to track waiting progress
                    if self.goal_wait_timer % 25 == 0:  # Print every ~0.4 seconds
                        print(
                            f"Waiting at Goal {self.gi}... {remaining_time:.1f}s remaining ({self.goal_wait_timer}/{self.goal_wait_steps} steps)"
                        )
                    # Ensure robot stays completely stationary while waiting
                    if self.use_physics_movement:
                        self.body.setVelocity([0, 0, 0])
                    # Lock position to prevent any drift during waiting
                    if self.goal_wait_position:
                        self._set_pose(
                            self.goal_wait_position[0],
                            self.goal_wait_position[1],
                            self.goal_wait_angle,
                        )
                    continue  # Skip all movement logic below

            # If we're currently at a goal but distance increased slightly due to movement inaccuracy,
            # don't reset the goal state - allow some tolerance
            elif self.at_goal and dist <= effective_stop_eps + 0.1:  # 10cm tolerance
                # Stay at goal, continue waiting, don't reset
                self.goal_wait_timer += 1
                if self.goal_wait_timer >= self.goal_wait_steps:
                    # Same completion logic as above
                    old_gi = self.gi
                    next_gi = (self.gi + 1) % len(self.goals)

                    if (
                        next_gi == self.mission_start_gi
                        and self.gi != self.mission_start_gi
                    ):
                        self.all_goals_completed = True
                        print(
                            f"🎉 Mission Complete! All {len(self.goals)} goals have been reached."
                        )
                        print("Robot stopping at final goal location.")
                        if self.use_physics_movement:
                            self.body.setVelocity([0, 0, 0])
                        # Don't return - let the mission completion logic at start of loop handle it
                        continue
                    else:
                        new_goal_x, new_goal_y = self.goals[next_gi]
                        print(
                            f"Goal {old_gi} completed! Moving to goal {next_gi} at ({new_goal_x:.2f}, {new_goal_y:.2f})"
                        )
                        print(
                            f"Robot current position when transitioning: ({x:.2f}, {y:.2f})"
                        )

                        self.gi = next_gi
                        self.at_goal = False
                        self.goal_wait_timer = 0
                        self.goal_wait_position = None  # Reset locked position
                        self.goal_wait_angle = 0.0  # Reset locked orientation
                        self.avoidance_timer = 0
                        self.backup_timer = 0
                        self.corner_escape_timer = 0
                        self.stuck_counter = 0
                        self.exploration_mode = False
                        self.bfs_path = []
                        self.bfs_target_index = 0
                        self.bfs_stuck_counter = 0
                        # Reset goal-oriented movement flags for new goal
                        self.is_facing_goal = False
                        self.reorient_after_avoidance = False
                        self.current_orientation_target = None

                        # Check if robot is in a problematic position after goal completion
                        if self._is_in_corner(x, y):
                            print(
                                "Robot seems trapped after goal completion, initiating escape sequence"
                            )
                            self.backup_timer = 8  # Reduced backup movement (16cm)
                            self.reorient_after_avoidance = True
                else:
                    remaining_time = (
                        (self.goal_wait_steps - self.goal_wait_timer) * self.dt / 1000.0
                    )
                    if self.goal_wait_timer % 25 == 0:
                        print(
                            f"Waiting at Goal {self.gi}... {remaining_time:.1f}s remaining ({self.goal_wait_timer}/{self.goal_wait_steps} steps)"
                        )
                    if self.use_physics_movement:
                        self.body.setVelocity([0, 0, 0])
                    # Lock position to prevent drift
                    if self.goal_wait_position:
                        self._set_pose(
                            self.goal_wait_position[0],
                            self.goal_wait_position[1],
                            self.goal_wait_angle,
                        )
                    continue
            else:
                # Truly not at goal - reset waiting state only if we're far enough away
                if (
                    not self.at_goal or dist > effective_stop_eps + 0.15
                ):  # 15cm reset threshold
                    self.at_goal = False
                    self.goal_wait_timer = 0
                    self.goal_wait_position = None  # Reset locked position
                    self.goal_wait_angle = 0.0  # Reset locked orientation
                    if dist > effective_stop_eps:
                        print(
                            f"Current pos: ({x:.2f}, {y:.2f}), Goal {self.gi} pos: ({self.goals[self.gi][0]:.2f}, {self.goals[self.gi][1]:.2f}), Distance to goal {self.gi}: {dist:.2f}, Threshold: {effective_stop_eps:.2f}"
                        )

            # Only proceed with movement if definitely not at goal
            if not self.at_goal:
                if self.use_physics_movement:
                    # For physics-based movement, calculate desired direction and let physics handle collision
                    dx, dy = tx - x, ty - y
                    dist = math.hypot(dx, dy)

                    # Calculate desired direction to goal (needed for all movement modes)
                    goal_angle = math.atan2(dy, dx)
                    movement_angle = goal_angle

                    # Goal-oriented movement: First check if robot should orient toward goal
                    if self.goal_oriented_mode and not self.exploration_mode:
                        # Check if robot is facing the goal
                        self.is_facing_goal = self._is_facing_goal(x, y, tx, ty)

                        # If not facing goal and not in active avoidance/escape mode, orient first
                        if (
                            not self.is_facing_goal
                            and self.avoidance_timer <= 0
                            and self.backup_timer <= 0
                            and self.corner_escape_timer <= 0
                        ):

                            print(f"Orienting toward Goal {self.gi}")
                            self._orient_toward_goal(x, y, tx, ty)
                            # Don't move this step, just orient
                            if self.use_physics_movement:
                                self.body.setVelocity([0, 0, 0])
                            continue

                        # If just finished obstacle avoidance, reorient toward goal
                        elif (
                            self.reorient_after_avoidance
                            and self.avoidance_timer <= 0
                            and self.backup_timer <= 0
                            and self.corner_escape_timer <= 0
                        ):

                            print(
                                f"Reorienting toward Goal {self.gi} after obstacle avoidance"
                            )
                            self._orient_toward_goal(x, y, tx, ty)
                            self.reorient_after_avoidance = False
                            # Don't move this step, just orient
                            if self.use_physics_movement:
                                self.body.setVelocity([0, 0, 0])
                            continue

                    # Check if robot is stuck
                    is_stuck = self._is_stuck(x, y)
                    is_in_corner = self._is_in_corner(x, y)

                    # BFS exploration mode - when stuck, use BFS pathfinding
                    if (
                        is_stuck and self.stuck_counter > 10
                    ):  # Reduced threshold for faster BFS activation
                        if not self.exploration_mode:
                            print(
                                "Robot heavily stuck, switching to BFS exploration mode"
                            )
                            self.exploration_mode = True
                            self.bfs_path = []
                            self.bfs_target_index = 0

                        bfs_angle = self._execute_bfs_navigation(x, y, tx, ty)
                        if bfs_angle is not None:
                            movement_angle = bfs_angle
                        else:
                            # BFS failed, use backup behavior
                            movement_angle = self._get_backup_direction(tx, ty, x, y)
                    else:
                        # Normal navigation logic
                        self.exploration_mode = False

                    if dist > effective_stop_eps and not self.exploration_mode:
                        # Handle corner situations more aggressively
                        if is_stuck and is_in_corner:
                            self.corner_escape_timer = (
                                20  # Reduced corner escape (40cm)
                            )
                            self.backup_timer = 0  # Reset normal backup
                            self.stuck_counter = 0  # Reset stuck counter
                            self.avoidance_timer = 0  # Reset avoidance
                            self.reorient_after_avoidance = (
                                True  # Flag for reorientation after corner escape
                            )

                        elif is_stuck:
                            self.backup_timer = 8  # Reduced backup distance (16cm)
                            self.stuck_counter = 0  # Reset stuck counter
                            self.avoidance_timer = 0  # Reset avoidance
                            self.reorient_after_avoidance = (
                                True  # Flag for reorientation after backup
                            )

                    # Handle different movement modes (only when not in BFS exploration mode)
                    if not self.exploration_mode:
                        if self.corner_escape_timer > 0:
                            # Corner escape: try multiple directions systematically
                            escape_phase = (40 - self.corner_escape_timer) // 10
                            escape_angles = [
                                goal_angle + math.pi,  # Phase 0: Back up
                                goal_angle + math.pi / 2,  # Phase 1: Side step left
                                goal_angle - math.pi / 2,  # Phase 2: Side step right
                                goal_angle + 3 * math.pi / 4,  # Phase 3: Diagonal back
                            ]

                            if escape_phase < len(escape_angles):
                                movement_angle = escape_angles[escape_phase]
                            else:
                                movement_angle = goal_angle + math.pi  # Default backup

                            self.corner_escape_timer -= 1

                        elif self.backup_timer > 0:
                            # Normal backup with obstacle checking
                            movement_angle = self._get_backup_direction(tx, ty, x, y)
                            self.backup_timer -= 1

                        else:
                            # Normal obstacle avoidance
                            check_distance = min(STEP_M * 4, dist)  # Look further ahead
                            future_x = x + math.cos(goal_angle) * check_distance
                            future_y = y + math.sin(goal_angle) * check_distance

                            collision_ahead = self._check_collision_with_buffer(
                                [x, y], [future_x, future_y]
                            )

                            if collision_ahead:
                                # Start or continue obstacle avoidance
                                if self.avoidance_timer <= 0:
                                    self.avoidance_timer = self.max_avoidance_time
                                    self.reorient_after_avoidance = (
                                        True  # Flag for reorientation after avoidance
                                    )

                                # Find avoidance direction
                                movement_angle = self._find_avoidance_direction(
                                    x, y, tx, ty
                                )
                                self.avoidance_timer -= 1

                                # Only switch wall follow side if really stuck for a long time
                                if (
                                    self.avoidance_timer < 10
                                ):  # Reduced from 20 to 10, less switching
                                    # Don't immediately reset timer - let it run down to force other behaviors
                                    self.avoidance_timer = 0  # Force exit to stuck detection instead of infinite switching

                            else:
                                # Clear path to goal - aggressively reset avoidance
                                if self.avoidance_timer > 0:
                                    # Check if we're far enough from obstacles to go direct
                                    direct_clear = True
                                    for check_dist in [STEP_M, STEP_M * 2, STEP_M * 3]:
                                        test_x = x + math.cos(goal_angle) * check_dist
                                        test_y = y + math.sin(goal_angle) * check_dist
                                        if self._check_collision_with_buffer(
                                            [x, y], [test_x, test_y]
                                        ):
                                            direct_clear = False
                                            break

                                    if direct_clear:
                                        # Completely clear path - reset avoidance immediately
                                        self.avoidance_timer = 0
                                        # Don't clear reorient flag here - let natural reorientation logic handle it
                                        movement_angle = goal_angle
                                    else:
                                        # Still some obstacles - gradual transition
                                        self.avoidance_timer -= 3  # Faster decay
                                        if self.avoidance_timer < 15:
                                            blend = 1.0 - (self.avoidance_timer / 15.0)
                                            current_avoid = (
                                                self._find_avoidance_direction(
                                                    x, y, tx, ty
                                                )
                                            )
                                            movement_angle = (
                                                current_avoid * (1 - blend)
                                                + goal_angle * blend
                                            )
                                        else:
                                            movement_angle = (
                                                self._find_avoidance_direction(
                                                    x, y, tx, ty
                                                )
                                            )

                                if self.avoidance_timer <= 0:
                                    movement_angle = goal_angle  # Direct to goal

                    # Calculate target position for this step
                    step = min(STEP_M, dist)
                    target_x = x + math.cos(movement_angle) * step
                    target_y = y + math.sin(movement_angle) * step

                    # Final collision check and move (with buffer)
                    if not self._check_collision_with_buffer(
                        [x, y], [target_x, target_y]
                    ):
                        self._set_pose(target_x, target_y, movement_angle)
                    else:
                        # Can't move, will be detected as stuck soon
                        self.body.setVelocity([0, 0, 0])

                    # Note: Goal completion is handled by the main goal waiting logic above
                else:
                    # Original direct position control method
                    # print(f"Using direct position control")
                    x, y, yaw, hit_or_arrived = self._advance_toward(x, y, tx, ty)
                    self._set_pose(x, y, yaw)

                    # Note: Goal completion is handled by the main goal waiting logic above


if __name__ == "__main__":
    PedBody().run()
