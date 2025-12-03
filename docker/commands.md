# System Requirements
- MacOS system
- [Install Webots](https://cyberbotics.com/doc/guide/installation-procedure#from-the-installation-file)
- Install python

# On your local machine

## Initial one-time setup
Clone this repo
```
git clone https://github.com/IshitaBadole/hri-goal-inference.git
```

Create a Docker image from the Dockerfile
```
cd hri-goal-inference/docker
docker build -t ros2-humble-webots .
```

Create and run a new container "ros2-webots-dev" from the image.
Mount `~/webots_shared` (Mac) to `/root/shared` (container).
Mount `~/webots_server` (Mac) to `/root/webots_server` (container) for persistent logs.
Set the environment variable for webots-server.
```
docker run -it --name ros2-webots-dev \
  -v $HOME/webots_shared:/root/shared \
  -v $HOME/webots_server:/root/webots_server \
  -e WEBOTS_SHARED_FOLDER="$HOME/webots_shared:/root/shared" \
  ros2-humble-webots`
```

## After Initial Setup
### Terminal 1: Run the local simulation server script
```
cd ~/webots_shared
export WEBOTS_HOME=/Applications/Webots.app
curl -O https://raw.githubusercontent.com/cyberbotics/webots-server/master/local_simulation_server.py
python3 local_simulation_server.py
```

### Terminal 2: Launch Webots from inside the container

Start and enter the container
```
docker start ros2-webots-dev
docker exec -it ros2-webots-dev bash
```

Clone the repo (skip if already done before)
```
git clone https://github.com/IshitaBadole/hri-goal-inference.git
```

[TEMP] Checkout to the teleop branch
```
git checkout pedestrian-teleop
```

Build the package
```
cd hri-goal-inference/ros2_ws
colcon build
source /opt/ros/humble/setup.bash
source install/local_setup.bash
```

Launch the default world (arena with pedestrian and simple obstacles)
```
ros2 launch my_package robot_launch.py
```

You can also run the launch commands with the world argument
```
ros2 launch my_package robot_launch.py world:=apartment.wbt

ros2 launch my_package robot_launch.py world:=break_room.wbt

ros2 launch my_package robot_launch.py world:=factory.wbt

ros2 launch my_package robot_launch.py world:=hall.wbt
```

WeBots will launch with the corresponding world. Keep this terminal running.

### Terminal 3: Run teleop script inside the container

Start and enter the container
```
docker start ros2-webots-dev
docker exec -it ros2-webots-dev bash
```

```
cd hri-goal-inference/ros2_ws
colcon build
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 run my_package teleop
```

> **Teleop Instructions**
> Keep focus in the teleop terminal by clicking in it. Use the teleop keys to move the pedestrian around in Webots. For best experience, drag the Webots window and teleop terminal on the same screen. Ensure that the focused window is the teleop terminal.

# Development and debugging commands

Check if anything is being published on /cmd_vel
```
ros2 topic echo /cmd_vel
```

See publishers/subscribers of a topic
```
ros2 topic info /cmd_vel
``` 

Check ros2 running nodes
```
ros2 node list
```

Check driver's subscribed topic (pedestrian driver)
```
ros2 node info /pedestrian_robot_driver
```

Stop and remove the container
```
docker stop ros2-webots-dev
docker rm ros2-webots-dev
```

Update repo to latest code and switch to your branch (run inside container)
```shell
cd hri-goal-inference
git fetch origin
git checkout pedestrian-teleop
git pull origin pedestrian-teleop
```

