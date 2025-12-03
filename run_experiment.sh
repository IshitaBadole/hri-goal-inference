#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: ./run_experiment.sh <participant_id> <world> <scenario_no>"
    echo ""
    echo "Example: ./run_experiment.sh p1 apartment.wbt 1"
    echo ""
    echo "Available worlds:"
    echo "  - apartment.wbt"
    echo "  - break_room.wbt"
    echo "  - factory.wbt"
    echo "  - hall.wbt"
    echo "  - my_world.wbt (default)"
    exit 1
fi

PARTICIPANT_ID=$1
WORLD=$2
SCENARIO_NO=$3

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed."
    echo "Install it with: brew install tmux"
    exit 1
fi

# Check if Docker container exists
if ! docker ps -a --format '{{.Names}}' | grep -q '^ros2-webots-dev$'; then
    echo "ERROR: Docker container 'ros2-webots-dev' not found."
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Check if webots_shared directory exists
if [ ! -d "$HOME/webots_shared" ]; then
    echo "ERROR: $HOME/webots_shared directory not found."
    echo "Please run ./setup.sh first."
    exit 1
fi

# Check if local_simulation_server.py exists
if [ ! -f "$HOME/webots_shared/local_simulation_server.py" ]; then
    echo "ERROR: local_simulation_server.py not found in $HOME/webots_shared"
    echo "Download it with:"
    echo "  cd $HOME/webots_shared"
    echo "  curl -O https://raw.githubusercontent.com/cyberbotics/webots-server/master/local_simulation_server.py"
    exit 1
fi

# Set WEBOTS_HOME (adjust if needed)
WEBOTS_HOME="${WEBOTS_HOME:-/Applications/Webots.app}"
if [ ! -d "$WEBOTS_HOME" ]; then
    echo "WARNING: Webots not found at $WEBOTS_HOME"
    echo "Set WEBOTS_HOME environment variable to your Webots installation path"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

SESSION_NAME="hri_experiment_${PARTICIPANT_ID}_${SCENARIO_NO}"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

echo "=========================================="
echo "Starting HRI Goal Inference Experiment"
echo "=========================================="
echo "Participant: $PARTICIPANT_ID"
echo "World: $WORLD"
echo "Scenario: $SCENARIO_NO"
echo "Session: $SESSION_NAME"
echo ""
echo "Creating tmux session with 3 terminals..."
echo ""

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "webots-server"

# Window 0: Webots Server (Mac)
tmux send-keys -t "$SESSION_NAME:0" "cd $HOME/webots_shared" C-m
tmux send-keys -t "$SESSION_NAME:0" "export WEBOTS_HOME=$WEBOTS_HOME" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo 'Starting Webots Server...'" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo 'Press Ctrl+C to stop when experiment is done.'" C-m
tmux send-keys -t "$SESSION_NAME:0" "python3 local_simulation_server.py" C-m

# Window 1: ROS Launcher (Container)
tmux new-window -t "$SESSION_NAME:1" -n "ros-launcher"
tmux send-keys -t "$SESSION_NAME:1" "docker start ros2-webots-dev" C-m
tmux send-keys -t "$SESSION_NAME:1" "sleep 2" C-m
tmux send-keys -t "$SESSION_NAME:1" "docker exec -it ros2-webots-dev bash -c 'cd hri-goal-inference/ros2_ws && source /opt/ros/humble/setup.bash && source install/local_setup.bash && ros2 launch my_package robot_launch.py participant_id:=$PARTICIPANT_ID world:=$WORLD scenario_no:=$SCENARIO_NO'" C-m

# Window 2: Teleop (Container)
tmux new-window -t "$SESSION_NAME:2" -n "teleop"
tmux send-keys -t "$SESSION_NAME:2" "echo 'Waiting for ROS launcher to start...'" C-m
tmux send-keys -t "$SESSION_NAME:2" "sleep 8" C-m
tmux send-keys -t "$SESSION_NAME:2" "docker exec -it ros2-webots-dev bash -c 'cd hri-goal-inference/ros2_ws && source /opt/ros/humble/setup.bash && source install/local_setup.bash && ros2 run my_package teleop'" C-m

echo "=========================================="
echo "✓ Experiment session started!"
echo "=========================================="
echo ""
echo "Tmux commands:"
echo "  - Switch windows: Ctrl+b then 0/1/2"
echo "  - Scroll in window: Ctrl+b then ["
echo "  - Detach session: Ctrl+b then d"
echo "  - Reattach session: tmux attach -t $SESSION_NAME"
echo ""
echo "To stop the experiment (in order):"
echo "  1. Stop teleop (window 2): Ctrl+C"
echo "  2. Stop ROS launcher (window 1): Ctrl+C"
echo "  3. Stop webots-server (window 0): Ctrl+C"
echo "  4. Exit session: Ctrl+b then :kill-session"
echo ""
echo "Logs will be saved to: $HOME/webots_server/pedestrian_logs/"
echo "Log file: ${PARTICIPANT_ID}_${WORLD%.wbt}_${SCENARIO_NO}.csv"
echo ""
echo "Attaching to tmux session..."
sleep 2

# Attach to the session (start on teleop window)
tmux select-window -t "$SESSION_NAME:2"
tmux attach-session -t "$SESSION_NAME"
