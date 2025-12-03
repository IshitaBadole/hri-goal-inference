# hri-goal-inference

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

Build the ROS2-Humble Docker image from the Dockerfile
```
cd hri-goal-inference/docker
docker build -t ros2-humble-webots .
```

Create `webots_shared` and `webots_server` directories in the root directory.
```
cd ~
mkdir webots_shared
mkdir webots_server
```

Create and run a new container `ros2-webots-dev` from the image.
Mount `~/webots_shared` (Mac) to `/root/shared` (container).
Mount `~/webots_server` (Mac) to `/root/webots_server` (container) for persistent logs.
Set the environment variable for webots-server.
```
docker run -it --name ros2-webots-dev \
  -v $HOME/webots_shared:/root/shared \
  -v $HOME/webots_server:/root/webots_server \
  -e WEBOTS_SHARED_FOLDER="$HOME/webots_shared:/root/shared" \
  ros2-humble-webots
```

Start and enter the container
```
docker start ros2-webots-dev
docker exec -it ros2-webots-dev bash
```

Clone the repo inside the container. This is required because the repo contains all the launchers and modules associated with the ROS2 package.
```
git clone https://github.com/IshitaBadole/hri-goal-inference.git
```

## After Initial Setup
The experiment runs by having three terminals running simultaneously.

### Terminal 1: Run the local simulation server script on your local machine (NOT inside container)
```
cd ~/webots_shared
export WEBOTS_HOME=/Applications/Webots.app
curl -O https://raw.githubusercontent.com/cyberbotics/webots-server/master/local_simulation_server.py
python3 local_simulation_server.py
```
NOTE : Change the WEBOTS_HOME variable if the directory in which Webots is installed is different.

### Terminal 2: Launch Webots from inside the container

Start and enter the container
```
docker start ros2-webots-dev
docker exec -it ros2-webots-dev bash
```

Build the package
```
cd hri-goal-inference/ros2_ws
colcon build
source /opt/ros/humble/setup.bash
source install/local_setup.bash
```

Run the launch commands with the world, participant_id and scenario_no arguments as follows:
```
# Run experiment with break room world, participant p1, and scenario 1
ros2 launch my_package robot_launch.py world:=break_room.wbt participant_id:=p1 scenario_no:=1
```

The available world arguments are:
- `apartment.wbt`
- `break_room.wbt`
- `factory.wbt`

ROS2 will launch the corresponding world in WeBots and the logs will get generated for the correponding participant and scenario. Keep this terminal running.

### Terminal 3: Run teleop script inside the container

Start and enter the container
```
docker start ros2-webots-dev
docker exec -it ros2-webots-dev bash
```

Build the package
```
cd hri-goal-inference/ros2_ws
colcon build
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 run my_package teleop
```

> **Teleop Instructions**
> Keep focus in the teleop terminal by clicking in it. Use the teleop keys to move the pedestrian around in Webots. For best experience, drag the Webots window and teleop terminal on the same screen. Ensure that the focused window is the teleop terminal.

# End the experiment

Press `Ctrl+C` in Terminal 3 where teleop module is running to stop the teleop.

Press `Ctrl+C` in Terminal 2 where the pedestrain driver is running to close WeBots.

Press `Ctrl+C` in Terminal 1 to stop the local simulation server.

# Check experiment logs

On your local machine
```
cd ~/webots_server/pedestrian_logs
ls
```

You will see the log files for the participant id and scenario no. you passed in the launch arguments.

The log file name format is: `<participant_id>_<world>_<scenario_no>.csv ` (ex. p1_apartment_s1.csv)

View the logs
```
cat <log file name>
```

# Run GMM

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

Update repo to latest code (run inside container)
```shell
cd hri-goal-inference
git pull origin main
```

# WeBots setup

WeBots Version: R2022B

World Files: https://github.com/cyberbotics/webots/tree/909a02174b1eb83373a924c263da6e33d3921f35/projects/samples/environments/indoor/worlds

## AI Use Acknowledgement

This project was developed with assistance from GitHub Copilot, Claude Sonnet 4.5 (Anthropic) and ChatGPT GPT5-Pro (OpenAI). AI tools were used for:
- Code generation and debugging assistance
- ROS2 architecture and integration guidance  
- Docker configuration and setup
- Documentation structure and improvement suggestions
- Problem-solving and troubleshooting technical issues

All AI-generated code was reviewed, tested, and adapted to meet project requirements. The core research direction, experimental design, and algorithmic approach remain the original work of the project team.
