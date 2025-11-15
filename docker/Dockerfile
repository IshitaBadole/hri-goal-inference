FROM ros:humble-ros-base

# Install ROS 2 desktop tools, webots_ros2, colcon, git
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-webots-ros2 \
    python3-colcon-common-extensions \
    nano \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use bash as shell so we can source setup.bash
SHELL ["/bin/bash", "-c"]

# Auto-source ROS 2 when opening a shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /root