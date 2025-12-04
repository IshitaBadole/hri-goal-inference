#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "HRI Goal Inference - One-Time Setup"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This setup script is designed for macOS only."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker Desktop for Mac first."
    echo "Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Webots is installed
WEBOTS_PATH="/Applications/Webots.app"
if [ ! -d "$WEBOTS_PATH" ]; then
    echo "WARNING: Webots not found at $WEBOTS_PATH"
    echo "If Webots is installed elsewhere, you can continue and set WEBOTS_HOME manually later."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please install Webots from: https://cyberbotics.com/doc/guide/installation-procedure"
        exit 1
    fi
else
    echo "✓ Webots found at $WEBOTS_PATH"
fi
echo "✓ Docker found"
echo ""

# Create required directories on Mac
echo "Creating required directories..."
mkdir -p "$HOME/webots_shared"
mkdir -p "$HOME/webots_server/pedestrian_logs"
echo "✓ Created $HOME/webots_shared"
echo "✓ Created $HOME/webots_server/pedestrian_logs"
echo ""

# Setup Python virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Created virtual environment"
else
    echo "✓ Virtual environment already exists"
fi

echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Python dependencies installed"
deactivate
echo ""

# Build Docker image
echo "Building Docker image (this may take several minutes)..."
cd docker
docker build -t ros2-humble-webots .
echo "✓ Docker image 'ros2-humble-webots' built successfully"
echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q '^ros2-webots-dev$'; then
    echo "WARNING: Container 'ros2-webots-dev' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping and removing existing container..."
        docker stop ros2-webots-dev 2>/dev/null || true
        docker rm ros2-webots-dev
        echo "✓ Removed existing container"
    else
        echo "Keeping existing container. Setup complete."
        exit 0
    fi
fi

# Create Docker container with volume mounts
echo "Creating Docker container 'ros2-webots-dev'..."
docker run -it --name ros2-webots-dev \
  -v "$HOME/webots_shared:/root/shared" \
  -v "$HOME/webots_server:/root/webots_server" \
  -e WEBOTS_SHARED_FOLDER="$HOME/webots_shared:/root/shared" \
  ros2-humble-webots bash -c "
    echo 'Container created. Cloning repository...'
    git clone https://github.com/IshitaBadole/hri-goal-inference.git
    cd hri-goal-inference/ros2_ws
    echo 'Building ROS2 package...'
    colcon build
    echo '✓ ROS2 package built successfully'
    echo ''
    echo 'Setup complete inside container!'
  "

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download webots-server script to $HOME/webots_shared:"
echo "   cd $HOME/webots_shared"
echo "   export WEBOTS_HOME=/Applications/Webots.app"
echo "   curl -O https://raw.githubusercontent.com/cyberbotics/webots-server/master/local_simulation_server.py"
echo ""
echo "2. Update repository inside container to latest code:"
echo "   docker start ros2-webots-dev"
echo "   docker exec -it ros2-webots-dev bash -c 'cd hri-goal-inference && git pull && cd ros2_ws && colcon build'"
echo ""
echo "3. To run experiments, you'll need 3 terminals:"
echo ""
echo "   Terminal 1 (Mac): Start webots-server"
echo "     cd $HOME/webots_shared"
echo "     python3 local_simulation_server.py"
echo ""
echo "   Terminal 2 (Container): Launch simulation"
echo "     docker start ros2-webots-dev"
echo "     docker exec -it ros2-webots-dev bash"
echo "     cd hri-goal-inference/ros2_ws"
echo "     source /opt/ros/humble/setup.bash"
echo "     source install/local_setup.bash"
echo "     ros2 launch my_package robot_launch.py participant_id:=p1 world:=apartment.wbt scenario_no:=1"
echo ""
echo "   Terminal 3 (Container): Run teleop"
echo "     docker exec -it ros2-webots-dev bash"
echo "     cd hri-goal-inference/ros2_ws"
echo "     source /opt/ros/humble/setup.bash"
echo "     source install/local_setup.bash"
echo "     ros2 run my_package teleop"
echo ""
echo "Logs will be saved to: $HOME/webots_server/pedestrian_logs/"
echo ""
echo "For detailed instructions, see: README.md"
echo "=========================================="
