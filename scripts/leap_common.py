#!/usr/bin/env python3
import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from kortex_driver.msg import TwistCommand


def leap_to_arm_workspace(hand_pose, arm_ws_min, arm_ws_max, hand_ws_min, hand_ws_max):
    '''Transform the hand position to the robot workspace'''

    # Create a new Pose object to avoid modifying the input in-place
    new_pose = Pose()
    new_pose.orientation = hand_pose.orientation

    # Convert Leap Motion position to normalised range [0, 1]
    x_norm = (hand_pose.position.x - hand_ws_min[0]) / (hand_ws_max[0] - hand_ws_min[0])
    y_norm = (hand_pose.position.y - hand_ws_min[1]) / (hand_ws_max[1] - hand_ws_min[1])
    z_norm = (hand_pose.position.z - hand_ws_min[2]) / (hand_ws_max[2] - hand_ws_min[2])

    # Ensure the normalised position is within the range [0, 1]
    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    z_norm = np.clip(z_norm, 0.0, 1.0)

    # Apply linear scaling to the arm workspace
    new_pose.position.x = x_norm * (arm_ws_max[0] - arm_ws_min[0]) + arm_ws_min[0]
    new_pose.position.y = y_norm * (arm_ws_max[1] - arm_ws_min[1]) + arm_ws_min[1]
    new_pose.position.z = z_norm * (arm_ws_max[2] - arm_ws_min[2]) + arm_ws_min[2]

    return new_pose


def process_arm_state(data, base_link, ref_frame, tf_buffer):
        # Get the tool pose
        tool_pose = PoseStamped()
        tool_pose.header.frame_id = base_link
        tool_pose.pose.position.x = data.base.tool_pose_x
        tool_pose.pose.position.y = data.base.tool_pose_y
        tool_pose.pose.position.z = data.base.tool_pose_z
        # Euler to quaternion
        q = tf.transformations.quaternion_from_euler(np.radians(data.base.tool_pose_theta_x), np.radians(data.base.tool_pose_theta_y), np.radians(data.base.tool_pose_theta_z))
        tool_pose.pose.orientation.x = q[0]
        tool_pose.pose.orientation.y = q[1]
        tool_pose.pose.orientation.z = q[2]
        tool_pose.pose.orientation.w = q[3]

        if base_link == ref_frame:
            return tool_pose
        else:
            return tool_pose_to_ref_frame(tool_pose, base_link, ref_frame, tf_buffer)


def process_gripper_state(data):
        raw_state = data.position[7]
        return 0.0 if raw_state < 0.1 else 1.0

    
def tool_pose_to_ref_frame(pose, arm_base_link, ref_frame, tf_buffer):
    '''Transform the arm state to the reference frame'''
    # Get the transform from the arm base to the world frame
    try:
        transform = tf_buffer.lookup_transform(ref_frame, arm_base_link, rospy.Time(0), timeout=rospy.Duration(0.1))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr("Error when transforming arm state to reference frame: " + str(e))
        return None
    # Transform the tool pose to the reference frame
    new_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
    new_pose.header.frame_id = ref_frame 
    
    return new_pose


def mismatch_hand_arm(hand_pose, tool_pose):
    '''Check if the hand pose is aligned with the arm state. Returns the euclidean distance between the two'''
    try:
        return np.sqrt((hand_pose.position.x - tool_pose.pose.position.x)**2 + 
                    (hand_pose.position.y - tool_pose.pose.position.y)**2 + 
                    (hand_pose.position.z - tool_pose.pose.position.z)**2)    
    except Exception as e:
        rospy.logerr("Error when calculating mismatch between hand and arm: " + str(e))
        return np.inf
    

def scale_vel_log(vel, vel_scale_factor=1.0):
    '''Logarithmically scale the velocity'''
    return np.sign(vel) * np.log(1 + abs(vel) * vel_scale_factor) / np.log(1 + vel_scale_factor)


def hand_pose_to_kinova_goal(hand_pose, tool_pose, pub_arm_goal,
                       base_frame, ref_frame, tf_buffer,
                       vel_scale_factor=1.0, max_linear_vel=1.0, max_angular_vel=1.57):
    '''Convert the hand pose goal to a kinova cartesian velocity command'''
    hand_pose_rf = PoseStamped()
    hand_pose_rf.header.frame_id = ref_frame
    hand_pose_rf.pose.position = hand_pose.position

    # If the base frame is not the reference frame, transform back to the arm base frame
    if base_frame != ref_frame:
        hand_pose_bf = tool_pose_to_ref_frame(hand_pose_rf, ref_frame, base_frame, tf_buffer)
        tool_pose_bf = tool_pose_to_ref_frame(tool_pose, ref_frame, base_frame, tf_buffer)
    else:
        hand_pose_bf = hand_pose_rf
        tool_pose_bf = tool_pose

    if hand_pose_bf is None or tool_pose_bf is None:
        rospy.logerr("Failed to transform the hand pose or the tool pose to the arm base frame")
        return None

    # Calculate the velocity command to move the arm to the hand position
    # Linear velocity in m/s
    pos_x_diff = (hand_pose_bf.pose.position.x - tool_pose_bf.pose.position.x)
    pos_y_diff = (hand_pose_bf.pose.position.y - tool_pose_bf.pose.position.y)
    pos_z_diff = (hand_pose_bf.pose.position.z - tool_pose_bf.pose.position.z)

    # Convert orientation from quaternion to euler angles
    q_hand = [hand_pose.orientation.x, hand_pose.orientation.y, hand_pose.orientation.z, hand_pose.orientation.w]
    hand_theta_x, hand_theta_y, hand_theta_z = tf.transformations.euler_from_quaternion(q_hand)

    # Add correction to the orientation of the hand
    hand_theta_x = - hand_theta_x 
    hand_theta_y = hand_theta_y + np.pi/2
    hand_theta_z = hand_theta_z + np.pi/2

    # Get enf effector orientation in euler
    q_tool = [tool_pose.pose.orientation.x, tool_pose.pose.orientation.y, tool_pose.pose.orientation.z, tool_pose.pose.orientation.w]
    tool_theta_x, tool_theta_y, tool_theta_z = tf.transformations.euler_from_quaternion(q_tool)
    
    theta_x_diff = np.arctan2(np.sin(hand_theta_y - tool_theta_x), np.cos(hand_theta_y - tool_theta_x)) 
    theta_y_diff = np.arctan2(np.sin(hand_theta_z - tool_theta_z), np.cos(hand_theta_z - tool_theta_z)) 
    theta_z_diff = -np.arctan2(np.sin(hand_theta_x - tool_theta_y), np.cos(hand_theta_x - tool_theta_y))

    # Detect singularity (hand_theta_x ~ ±90°)
    is_singularity = abs(abs(hand_theta_x)- np.pi/2) < 0.175  # within 10°

    if is_singularity:
        theta_x_diff = 0
        theta_y_diff = 0
        theta_z_diff = 0

    # Build velocity command
    arm_vel = TwistStamped()
    arm_vel.header.stamp = rospy.Time.now()
    arm_vel.twist.linear.x = np.clip(scale_vel_log(pos_x_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    arm_vel.twist.linear.y = np.clip(scale_vel_log(pos_y_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    arm_vel.twist.linear.z = np.clip(scale_vel_log(pos_z_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    arm_vel.twist.angular.x = np.clip(scale_vel_log(theta_x_diff), -max_angular_vel, max_angular_vel)
    arm_vel.twist.angular.y = np.clip(scale_vel_log(theta_y_diff), -max_angular_vel, max_angular_vel)
    arm_vel.twist.angular.z = scale_vel_log(theta_z_diff)

    if pub_arm_goal is not None:
        pub_arm_goal.publish(arm_vel)

    return arm_vel


def reach_arm_position(arm_vel_command, pub_arm_vel):
    '''Move the arm towards the desired pose'''
    # Build Kinova's velocity command
    kinova_arm_vel = TwistCommand()
    kinova_arm_vel.reference_frame = 0
    kinova_arm_vel.twist.linear_x = arm_vel_command.twist.linear.x
    kinova_arm_vel.twist.linear_y = arm_vel_command.twist.linear.y
    kinova_arm_vel.twist.linear_z = arm_vel_command.twist.linear.z
    kinova_arm_vel.twist.angular_x = arm_vel_command.twist.angular.x
    kinova_arm_vel.twist.angular_y = arm_vel_command.twist.angular.y
    kinova_arm_vel.twist.angular_z = arm_vel_command.twist.angular.z

    pub_arm_vel.publish(kinova_arm_vel)


def reach_gripper_position(relative_position, gripper_joint):
    '''Move the gripper to a relative position between 0 and 1'''
    max_absolute_pos = gripper_joint.max_bound()
    min_absolute_pos = gripper_joint.min_bound()
    try:
        pos = relative_position * (max_absolute_pos - min_absolute_pos) + min_absolute_pos
        val = gripper_joint.move(pos, wait=True)
        return val
    except:
        return False 
   
    
def gripper_binary_control(raw_pinch_state, gripper_state, gripper_joint, pinch_threshold=0.1):
    '''Control the gripper using binary control (open or close)'''
    pinch_state = None
    if raw_pinch_state <= pinch_threshold:
        pinch_state = 0.0
    elif raw_pinch_state >= (1 - pinch_threshold):
        pinch_state = 1.0

    if (pinch_state == 0.0 or pinch_state == 1.0) and gripper_state != pinch_state:
        rospy.loginfo("Changing gripper state to " + str(pinch_state))
        reach_gripper_position(pinch_state, gripper_joint)


def reach_named_position(moveit_arm_group, target):
    '''Go to a named target'''       
    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    moveit_arm_group.set_named_target(target)
    # Plan the trajectory
    success_flag, trajectory_message, planning_time, error_code = moveit_arm_group.plan()
    # Execute the trajectory and block while it's not finished
    return moveit_arm_group.execute(trajectory_message, wait=True)