# Kinova Gen3 Teleoperation

ROS package for teleoperating Kinova Gen3 arms with a Leap Motion controller.

## Overview
The package provides a single-arm and a dual-arm stack. `leap_teleop.py` maps one Leap hand to one arm and controls a Robotiq 2F-85 gripper if one is present. In the dual-arm stack, `leap_to_kinova_dual.py` converts the two hands into Cartesian velocity commands and `teleop_dual_controller.py` forwards them to both arms, handles the MoveIt named poses and reacts to inactivity.

## Prerequisites
- ROS Noetic
- Kinova Kortex driver (`kortex_driver`) for each arm
- MoveIt configuration for the arm or arms: planning group `arm` for the single-arm stack and `both_arms` for the dual-arm stack, with named poses for the home and ready positions
- [leap-motion-ros](https://github.com/casado-fdo/leap-motion-ros), publishing `/leapmotion/hands/{left,right}`
- Robotiq 2F-85 gripper (optional)

## Installation
Clone the repository into a catkin workspace and build:
```bash
cd ~/catkin_ws/src
git clone https://github.com/casado-fdo/kinova_gen3_teleop.git
cd ..
catkin_make
```

## Usage
### Single arm
Start the Kortex driver, MoveIt and the Leap Motion node, then launch:
```bash
roslaunch kinova_gen3_teleop leap_teleop.launch
```
[`cfg/default.yaml`](cfg/default.yaml) is an example for an arm called `my_gen3` with the `arm` planning group. The launch arguments are `config_file` (default [`cfg/default.yaml`](cfg/default.yaml)), `node_name` (default `leap_teleop`) and `robot_name` (default `my_gen3`, used as the node namespace). Set the `robot_name` argument and the `robot_name` entry in the YAML to the same driver namespace.

### Dual arm
The dual-arm stack needs a configuration file with the keys described in [Configuration](#configuration). [`cfg/default.yaml`](cfg/default.yaml) only covers the single-arm stack.
```bash
roslaunch kinova_gen3_teleop leap_teleop_dual.launch config_file:=/path/to/dual_config.yaml
```
The launch arguments are `config_file`, `node_name` (default `leap_teleop`) and `imitation_mode` (default `demo_teleop`). The launch sets `teleop_source` to `leap_teleop` for `demo_teleop` and to `lerobot` otherwise. With `inference` the Leap mapping node is skipped and the controller reads `/<robot>/lerobot/cartesian_velocity` instead. The parent `blueberry_driver/launch/full_leap_teleop.launch` includes this launch file with its own dual-arm configuration.

## ROS Interface
In the single-arm stack, `<robot>` is the `robot_name` parameter and `<side>` is `leap_hand`:

| Topic | Type | Description |
|---|---|---|
| `/leapmotion/hands/<side>` | `leap_motion_controller/Hand` | Leap hand pose (subscribed) |
| `/leapmotion/hands/<side>/pinch` | `sensor_msgs/Range` | Pinch distance, subscribed when the gripper is present |
| `/<robot>/base_feedback` | `kortex_driver/BaseCyclic_Feedback` | Arm feedback (subscribed) |
| `/<robot>/base_feedback/joint_state` | `sensor_msgs/JointState` | Gripper feedback (subscribed) |
| `/<robot>/in/cartesian_velocity` | `kortex_driver/TwistCommand` | Cartesian velocity command (published) |
| `/<robot>/base/clear_faults` | `kortex_driver/Base_ClearFaults` | Fault clearing service, called at startup |

In the dual-arm stack, `<robot>` is `robot1_name` or `robot2_name`:

| Topic | Type | Description |
|---|---|---|
| `/leapmotion/hands/{left,right}` | `leap_motion_controller/Hand` | Leap hand poses (subscribed by the mapping node) |
| `/<robot>/base_feedback` | `kortex_driver/BaseCyclic_Feedback` | Arm feedback, subscribed by the mapping node |
| `/<robot>/leap_teleop/cartesian_velocity` | `geometry_msgs/TwistStamped` | Commands from the mapping node to the controller |
| `/<robot>/lerobot/cartesian_velocity` | `geometry_msgs/TwistStamped` | Controller input when `imitation_mode:=inference` |
| `/<robot>/in/cartesian_velocity` | `kortex_driver/TwistCommand` | Cartesian velocity command (published by the controller) |
| `/<robot>/base/clear_faults` | `kortex_driver/Base_ClearFaults` | Fault clearing service, called at startup |

## Configuration
[`cfg/default.yaml`](cfg/default.yaml) contains the single-arm parameters:

| Parameter | Default | Description |
|---|---|---|
| `robot_name` | `my_gen3` | Arm namespace |
| `arm_group_name` | `arm` | MoveIt planning group |
| `leap_ns` | `/leapmotion` | Leap Motion topic namespace |
| `leap_hand` | `right` | Hand used, `left` or `right` |
| `arm_base_link`, `arm_reference_frame` | `base_link` | TF frames for the arm and the shared reference |
| `arm_workspace_min`, `arm_workspace_max` | see file | Workspace in the reference frame, m |
| `leap_workspace_min`, `leap_workspace_max` | see file | Leap workspace in the Leap frame, m |
| `moveit_max_acc`, `moveit_max_vel` | `0.2`, `0.3` | MoveIt scaling factors for named poses |
| `home_pose_name`, `ready_pose_name` | `retract`, `home` | MoveIt named poses |
| `vel_scale_factor` | `1.5` | Logarithmic velocity gain |
| `max_linear_vel`, `max_angular_vel` | `1.0`, `1.57` | Velocity limits, m/s and rad/s |
| `mismatch_threshold` | `0.5` | Maximum hand-arm distance before commands are ignored, m |
| `pinch_threshold` | `0.35` | Pinch threshold, fraction of the sensor range |
| `inactivity_timeout` | `5.0` | Seconds without commands before returning to the ready pose; `-1` disables it |
| `control_gripper` | `true` | Enable gripper control when the gripper is present |

The dual-arm stack uses the same MoveIt and teleoperation keys, with these differences and additions:

| Parameter | Description |
|---|---|
| `robot1_name`, `robot2_name` | Arm namespaces, for example `l_kinova_` and `r_kinova_` |
| `arm_group_name` | MoveIt planning group for both arms, for example `both_arms` |
| `arm1_base_link`, `arm2_base_link` | Base frame of each arm |
| `arm_ref_frame` | Common reference frame, replaces `arm_base_link` and `arm_reference_frame` |
| `arm1_workspace_min`, `arm1_workspace_max`, `arm2_workspace_min`, `arm2_workspace_max` | Per-arm workspace in `arm_ref_frame`, m |
| `l_hand_workspace_min`, `l_hand_workspace_max`, `r_hand_workspace_min`, `r_hand_workspace_max` | Per-hand workspace in the Leap frame, m |
| `home_after_idle_sec` | Seconds at the ready pose before going home; `-1` disables it (default) |


## License
This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Acknowledgments
Developed at the Personal Robotics Laboratory, Imperial College London.
