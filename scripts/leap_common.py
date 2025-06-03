#!/usr/bin/env python3
import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import PoseStamped
from kortex_driver.msg import TwistCommand


def leap_to_arm_workspace(hand_pose, arm_ws_min, arm_ws_max, hand_ws_min, hand_ws_max):
        '''Transform the hand position to the robot workspace'''

        # Convert Leap Motion position to normalised range [0, 1]
        hand_pose.position.x = (hand_pose.position.x - hand_ws_min[0]) / (hand_ws_max[0] - hand_ws_min[0])
        hand_pose.position.y = (hand_pose.position.y - hand_ws_min[1]) / (hand_ws_max[1] - hand_ws_min[1])
        hand_pose.position.z = (hand_pose.position.z - hand_ws_min[2]) / (hand_ws_max[2] - hand_ws_min[2])

        # Apply linear scaling to the arm workspace
        hand_pose.position.x = hand_pose.position.x * (arm_ws_max[0] - arm_ws_min[0]) + arm_ws_min[0]
        hand_pose.position.y = hand_pose.position.y * (arm_ws_max[1] - arm_ws_min[1]) + arm_ws_min[1]
        hand_pose.position.z = hand_pose.position.z * (arm_ws_max[2] - arm_ws_min[2]) + arm_ws_min[2]

        return hand_pose


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


def reach_arm_position(hand_pose, tool_pose, pub_arm_vel, vel_scale_factor=1.0, max_linear_vel=1.0, max_angular_vel=1.57):
    '''Move the arm to the hand position'''
    # Calculate the velocity command to move the arm to the hand position
    arm_vel = TwistCommand()
    arm_vel.reference_frame = 0
    # Linear velocity in m/s
    pos_x_diff = (hand_pose.position.x - tool_pose.pose.position.x)
    pos_y_diff = (hand_pose.position.y - tool_pose.pose.position.y)
    pos_z_diff = (hand_pose.position.z - tool_pose.pose.position.z)
    arm_vel.twist.linear_x = np.clip(scale_vel_log(pos_x_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    arm_vel.twist.linear_y = np.clip(scale_vel_log(pos_y_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    arm_vel.twist.linear_z = np.clip(scale_vel_log(pos_z_diff, vel_scale_factor), -max_linear_vel, max_linear_vel)
    
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
    
    theta_x_diff = hand_theta_y - tool_theta_x
    theta_y_diff = hand_theta_z - tool_theta_z
    theta_z_diff = - (hand_theta_x - tool_theta_y)
    arm_vel.twist.angular_x = np.clip(scale_vel_log(theta_x_diff), -max_angular_vel, max_angular_vel)
    arm_vel.twist.angular_y = np.clip(scale_vel_log(theta_y_diff), -max_angular_vel, max_angular_vel)
    arm_vel.twist.angular_z = scale_vel_log(theta_z_diff) # no need to clip since this rotation is alreadu super slow

    # Publish the velocity command
    pub_arm_vel.publish(arm_vel)


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