# Build ROS image (only once)

`docker build -t ros2-humble-webots .`

`docker run -it --name ros2-webots-dev \
  -v $HOME/webots_shared:/root/shared \
  -e WEBOTS_SHARED_FOLDER="$HOME/webots_shared:/root/shared" \
  ros2-humble-webots`

# Clone repo in container
`git clone https://github.com/IshitaBadole/hri-goal-inference.git`

# All other times
`cd ~/webots_shared`

`export WEBOTS_HOME=/Applications/Webots.app`

`curl -O https://raw.githubusercontent.com/cyberbotics/webots-server/master/local_simulation_server.py`

`python3 local_simulation_server.py`

# Start and enter the container
`docker start ros2-webots-dev`

`docker exec -it ros2-webots-dev bash`

# Update repo to latest code and switch to your branch (run inside container)
`cd hri-goal-inference`

`git fetch origin`

`git checkout pedestrian-teleop`

`git pull origin pedestrian-teleop`

# Terminal 1 inside container (To launch webots and world)
`cd hri-goal-inference/ros2_ws/src`

`colcon build`

`source /opt/ros/humble/setup.bash`

`source install/local_setup.bash`

`ros2 launch my_package robot_launch.py`

## To launch webots with world file argument
`ros2 launch my_package robot_launch.py world:=apartment.wbt`
`ros2 launch my_package robot_launch.py world:=break_room.wbt`
`ros2 launch my_package robot_launch.py world:=factory.wbt`
`ros2 launch my_package robot_launch.py world:=hall.wbt`

# Terminal 2 inside container (For Teleop)
`cd hri-goal-inference/ros2_ws/src`

`colcon build`

`source /opt/ros/humble/setup.bash`

`source install/local_setup.bash`

`ros2 run my_package teleop`

# Debug teleop

## Check if anything is being published on /cmd_vel
`ros2 topic echo /cmd_vel`

## see publishers/subscriber
`ros2 topic info /cmd_vel ` 

## Check ros2 running nodes
`ros2 node list`

## Check driver's subscribed topic
`ros2 node info /my_robot_driver`

## Check driver's subscribed topic (pedestrian driver)
`ros2 node info /pedestrian_robot_driver`
