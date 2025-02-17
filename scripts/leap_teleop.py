#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
import tf
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range, JointState
from kortex_driver.msg import *
from kortex_driver.srv import *
from control_msgs.msg import GripperCommandActionGoal
import moveit_commander
import moveit_msgs.msg
import numpy as np
import sys


MOVEIT_MAX_ACC = 0.2 # %
MOVEIT_MAX_VEL = 0.7 # %
VEL_SCALE_FACTOR = 2.0
MISMATCH_THRESHOLD = 0.5 # m in arm base frame
PINCH_THRESHOLD = 0.35 # where 0.5 means 50% of the range

class KinovaTeleop(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)
        rospy.loginfo("Starting Kinova Gen3 teleoperation node for " + self.robot_name)

        # Create the MoveItInterface necessary objects
        arm_group_name = rospy.get_param('arm_group_name')
        moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_arm_group = moveit_commander.MoveGroupCommander(arm_group_name)
        self.moveit_display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)
        # Limit velocity and acceleration
        self.moveit_arm_group.set_max_velocity_scaling_factor(MOVEIT_MAX_VEL)
        self.moveit_arm_group.set_max_acceleration_scaling_factor(MOVEIT_MAX_ACC)

        self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s

        # Ready and Home behaviours
        self.home_pose = rospy.get_param("home_pose_name")
        self.ready_pose = rospy.get_param("ready_pose_name")

        # Robot reference frame
        self.arm_base_link = rospy.get_param("arm_base_link")
        self.arm_ref_frame = rospy.get_param("arm_reference_frame")
        if self.arm_base_link != self.arm_ref_frame:
            rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        # Robot workspace boundaries
        self.arm_ws_min = rospy.get_param("arm_workspace_min")
        self.arm_ws_max = rospy.get_param("arm_workspace_max")

        # Leap motion workspace boundaries
        self.leap_ws_min = rospy.get_param("leap_workspace_min")
        self.leap_ws_max = rospy.get_param("leap_workspace_max")

        # Gripper setup
        self.control_gripper = rospy.get_param("control_gripper")
        self.is_gripper_present = rospy.get_param(self.robot_name + "/is_gripper_present", False)
        if self.control_gripper and self.is_gripper_present:
            gripper_joint_names = rospy.get_param(self.robot_name + "/gripper_joint_names", [])
            gripper_joint_name = gripper_joint_names[0]
            self.gripper_joint = self.moveit_robot.get_joint(gripper_joint_name)
            self.gripper_max_absolute_pos = self.gripper_joint.max_bound()
            self.gripper_min_absolute_pos = self.gripper_joint.min_bound()
            rospy.loginfo("Gripper control enabled")
        else:
            self.gripper_joint_name = ""
            rospy.loginfo("Gripper control disabled")
        self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())


        # Subscribe to the Leap data
        self.leap_ns = rospy.get_param('leap_ns', "/leapmotion")
        self.leap_hand = rospy.get_param("leap_hand", "right")
        hand_data = self.leap_ns + '/hands/' + self.leap_hand
        self.sub_hand_pose = rospy.Subscriber(hand_data + '/pose', PoseStamped, self.hand_pose_callback)
        self.sub_hand_pinch = rospy.Subscriber(hand_data + '/pinch', Range, self.hand_pinch_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + hand_data) 

        # Subscribe to the arm and gripper state
        self.sub_arm_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback', BaseCyclic_Feedback, self.arm_state_callback)
        self.sub_gripper_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback/joint_state', JointState, self.gripper_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot_name + "/base_feedback")

        # Create publisher for cartesian velocity commands
        self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        
        # Create publisher for gripper commands
        self.pub_gripper = rospy.Publisher('/' + self.robot_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)

        # Initialize the hand pose and arm state
        self.last_hand_pose= None
        self.last_pinch_state = 0.0
        self.last_gripper_state = None
        self.last_arm_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(50)

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    

    def signal_term_handler(self):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.loginfo('User Keyboard interrupt. Going home...')
        # Try to go home
        max_acc = MOVEIT_MAX_ACC
        max_vel = MOVEIT_MAX_VEL
        is_home = False
        while not is_home and max_acc >= 0.01:
            rospy.loginfo("Trying to go home (max_acc=" + str(max_acc) + ", max_vel=" + str(max_vel) + ")")
            is_home = self.go_to(self.home_pose)
            # Next time try with half the acceleration
            max_acc = max_acc / 2
            max_vel = max_vel / 2
            self.moveit_arm_group.set_max_acceleration_scaling_factor(max_acc)
            self.moveit_arm_group.set_max_velocity_scaling_factor(max_vel)

        if not is_home:
            rospy.logerr("Failed to go home")
        else:
            rospy.loginfo("Went home successfully")

        # Stop the arm
        rospy.loginfo('Stopping the arm and shutting down...')
        self.stop()


    def leap_to_arm_workspace(self, hand_pose):
        '''Transform the hand position to the robot workspace'''

        # Convert Leap Motion position to normalised range [0, 1]
        hand_pose.position.x = (hand_pose.position.x - self.leap_ws_min[0]) / (self.leap_ws_max[0] - self.leap_ws_min[0])
        hand_pose.position.y = (hand_pose.position.y - self.leap_ws_min[1]) / (self.leap_ws_max[1] - self.leap_ws_min[1])
        hand_pose.position.z = (hand_pose.position.z - self.leap_ws_min[2]) / (self.leap_ws_max[2] - self.leap_ws_min[2])

        # Apply linear scaling to the arm workspace
        hand_pose.position.x = hand_pose.position.x * (self.arm_ws_max[0] - self.arm_ws_min[0]) + self.arm_ws_min[0]
        hand_pose.position.y = hand_pose.position.y * (self.arm_ws_max[1] - self.arm_ws_min[1]) + self.arm_ws_min[1]
        hand_pose.position.z = hand_pose.position.z * (self.arm_ws_max[2] - self.arm_ws_min[2]) + self.arm_ws_min[2]

        return hand_pose

    
    def tool_pose_to_ref_frame(self, pose):
        '''Transform the arm state to the reference frame'''
        # Get the transform from the arm base to the world frame
        try:
            transform = self.tf_buffer.lookup_transform(self.arm_ref_frame, self.arm_base_link, rospy.Time(0), timeout=rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Error when transforming arm state to reference frame: " + str(e))
            return None
        # Transform the tool pose to the reference frame
        new_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
        new_pose.header.frame_id = self.arm_ref_frame 
        
        return new_pose
    

    def mismatch_hand_arm(self, hand_pose, tool_pose):
        '''Check if the hand pose is aligned with the arm state. Returns the euclidean distance between the two'''
        try:
            return np.sqrt((hand_pose.position.x - tool_pose.pose.position.x)**2 + 
                        (hand_pose.position.y - tool_pose.pose.position.y)**2 + 
                        (hand_pose.position.z - tool_pose.pose.position.z)**2)    
        except Exception as e:
            rospy.logerr("Error when calculating mismatch between hand and arm: " + str(e))
            return np.inf

    def scale_vel_log(self, vel):
        '''Logarithmically scale the velocity'''
        return np.sign(vel) * np.log(1 + abs(vel) * VEL_SCALE_FACTOR) / np.log(1 + VEL_SCALE_FACTOR)
    

    def reach_arm_position(self, hand_pose, tool_pose):
        '''Move the arm to the hand position'''
        # Calculate the velocity command to move the arm to the hand position
        arm_vel = TwistCommand()
        arm_vel.reference_frame = 0
        # Linear velocity in m/s
        pos_x_diff = (hand_pose.position.x - tool_pose.pose.position.x)
        pos_y_diff = (hand_pose.position.y - tool_pose.pose.position.y)
        pos_z_diff = (hand_pose.position.z - tool_pose.pose.position.z)
        arm_vel.twist.linear_x = np.clip(self.scale_vel_log(pos_x_diff), -self.max_linear_vel, self.max_linear_vel)
        arm_vel.twist.linear_y = np.clip(self.scale_vel_log(pos_y_diff), -self.max_linear_vel, self.max_linear_vel)
        arm_vel.twist.linear_z = np.clip(self.scale_vel_log(pos_z_diff), -self.max_linear_vel, self.max_linear_vel)
        
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
        arm_vel.twist.angular_x = np.clip(self.scale_vel_log(theta_x_diff), -self.max_angular_vel, self.max_angular_vel)
        arm_vel.twist.angular_y = np.clip(self.scale_vel_log(theta_y_diff), -self.max_angular_vel, self.max_angular_vel)
        arm_vel.twist.angular_z = self.scale_vel_log(theta_z_diff) # no need to clip since this rotation is alreadu super slow

        # Publish the velocity command
        self.pub_arm_vel.publish(arm_vel)

    
    def reach_gripper_position(self, relative_position):
        '''Move the gripper to a relative position between 0 and 1'''
        try:
            pos = relative_position* (self.gripper_max_absolute_pos - self.gripper_min_absolute_pos) + self.gripper_min_absolute_pos
            val = self.gripper_joint.move(pos, wait=True)
            return val
        except:
            return False 
        
    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm state to be received
        while self.last_arm_state is None:
            rospy.loginfo_throttle(10, "Waiting for arm state...")
            self.rate.sleep()
        # Go home
        if not self.go_to(self.ready_pose):
            rospy.logerr("Failed to go home")
            return
        rospy.loginfo("Ready to teleoperate the arm!")
        while not rospy.is_shutdown():
            # Get the current hand position and orientation
            hand_pose = self.last_hand_pose
            pinch_state = self.last_pinch_state

            # Check if the hand pose is received
            if hand_pose is None or pinch_state is None:
                rospy.loginfo_throttle(10, "Waiting for hand pose...")
                self.stop()
                self.rate.sleep()
                continue

            # Get the current arm and gripper states
            arm_state = self.last_arm_state
            gripper_state = self.last_gripper_state

            # Transform the hand position and orientation to the base frame
            hand_pose = self.leap_to_arm_workspace(hand_pose)

            # Check that the hand pose is aligned with the current arm state
            diff = self.mismatch_hand_arm(hand_pose, arm_state)
            if diff > MISMATCH_THRESHOLD:
                rospy.loginfo_throttle(1, "Waiting for hand pose to align with arm...")
                self.rate.sleep()
                self.stop()
                self.last_hand_pose = None
                continue

            # All good to send velocity commands
            rospy.loginfo_throttle(60, "Sending velocity commands to the arm...")

            # Control the arm
            self.reach_arm_position(hand_pose, arm_state)

            # Control the gripper
            if self.control_gripper and self.is_gripper_present:
                if pinch_state <= PINCH_THRESHOLD:
                    pinch_state = 0.0
                elif pinch_state >= (1 - PINCH_THRESHOLD):
                    pinch_state = 1.0
                if (pinch_state == 0.0 or pinch_state == 1.0) and gripper_state != pinch_state:
                    rospy.loginfo("Changing gripper state to " + str(pinch_state))
                    self.reach_gripper_position(pinch_state)

            # Clear the last hand pose
            self.last_hand_pose = None

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.pose
    

    def hand_pinch_callback(self, data):
        # Update the pinch state using a windowed average
        self.last_pinch_state = 0.99 * self.last_pinch_state + 0.01 * data.range


    def arm_state_callback(self, data):
        # Get the tool pose
        tool_pose = PoseStamped()
        tool_pose.header.frame_id = self.arm_base_link
        tool_pose.pose.position.x = data.base.tool_pose_x
        tool_pose.pose.position.y = data.base.tool_pose_y
        tool_pose.pose.position.z = data.base.tool_pose_z
        # Euler to quaternion
        q = tf.transformations.quaternion_from_euler(np.radians(data.base.tool_pose_theta_x), np.radians(data.base.tool_pose_theta_y), np.radians(data.base.tool_pose_theta_z))
        tool_pose.pose.orientation.x = q[0]
        tool_pose.pose.orientation.y = q[1]
        tool_pose.pose.orientation.z = q[2]
        tool_pose.pose.orientation.w = q[3]

        if self.arm_base_link == self.arm_ref_frame:
            self.last_arm_state = tool_pose
        else:
            # Transform to the arm base frame
            self.last_arm_state = self.tool_pose_to_ref_frame(tool_pose)            

    
    def gripper_state_callback(self, data):
        # Get the current gripper state
        raw_state = data.position[self.degrees_of_freedom]
        self.last_gripper_state = 0.0 if raw_state < 0.1 else 1.0


    def action_topic_callback(self, notif):
        self.last_action_notif_type = notif.action_event 


    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                rospy.loginfo("Waiting for action to end or abort...")
                self.rate.sleep()


    def reach_named_position(self, target):
        '''Go to a named target'''       
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        self.moveit_arm_group.set_named_target(target)
        # Plan the trajectory
        success_flag, trajectory_message, planning_time, error_code = self.moveit_arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return self.moveit_arm_group.execute(trajectory_message, wait=True)


    def go_to(self, pose_name, open_gripper=True):
        '''Go to the home position'''
        reach_pose = self.reach_named_position(pose_name)
        if open_gripper and self.control_gripper and self.is_gripper_present:
            open = self.reach_gripper_position(0.0)
        else:
            open = True
        return reach_pose and open
    
    def stop(self):
        arm_vel = TwistCommand()
        arm_vel.reference_frame = 0
        arm_vel.twist.linear_x = 0
        arm_vel.twist.linear_y = 0
        arm_vel.twist.linear_z = 0
        arm_vel.twist.angular_x = 0
        arm_vel.twist.angular_y = 0
        arm_vel.twist.angular_z = 0
        self.pub_arm_vel.publish(arm_vel)


if __name__ == '__main__':
    teleop = KinovaTeleop()
    teleop.run()
