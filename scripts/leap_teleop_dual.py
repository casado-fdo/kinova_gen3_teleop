#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
from sensor_msgs.msg import Range, JointState
from leap_motion_controller.msg import Hand
from kortex_driver.msg import BaseCyclic_Feedback, ActionEvent, TwistCommand
from kortex_driver.srv import *
from control_msgs.msg import GripperCommandActionGoal
import moveit_commander
import moveit_msgs.msg
import numpy as np
import sys
import leap_common as lc


class KinovaTeleopDual(object):
    def __init__(self):
        # Initialize the node
        self.robot1_name = rospy.get_param('robot1_name')
        self.robot2_name = rospy.get_param('robot2_name')
        rospy.init_node('kinova_teleop_dual', anonymous=True)
        rospy.loginfo("Starting Dual Kinova Gen3 teleoperation node")

        # Get MoveIt! parameters
        self.home_pose = rospy.get_param("home_pose_name")
        self.ready_pose = rospy.get_param("ready_pose_name")
        self.moveit_max_vel = rospy.get_param("moveit_max_vel")
        self.moveit_max_acc = rospy.get_param("moveit_max_acc")
        arm_group_name = rospy.get_param('arm_group_name')

        # Get teleoperation parameters
        self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m
        self.pinch_threshold = rospy.get_param("pinch_threshold") # m [0-1]
        self.control_gripper_right = rospy.get_param("control_gripper_right") # true/false
        self.control_gripper_left = rospy.get_param("control_gripper_left")
        self.inactivity_timeout = rospy.get_param("inactivity_timeout") # s (-1 for infinite)
        self.last_command_time = None

        # Create the MoveItInterface necessary objects
        moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_arm_group = moveit_commander.MoveGroupCommander(arm_group_name)
        self.moveit_display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)
        self.moveit_arm_group.set_max_velocity_scaling_factor(self.moveit_max_vel)
        self.moveit_arm_group.set_max_acceleration_scaling_factor(self.moveit_max_acc)

        # Robot reference frame
        self.arm1_base_link = rospy.get_param("arm1_base_link")
        self.arm2_base_link = rospy.get_param("arm2_base_link")
        self.reference_frame = rospy.get_param("reference_frame")
        rospy.loginfo("Transforming arm state to reference frame " + self.reference_frame)

        # Robot workspace boundaries
        self.arm1_ws_min = rospy.get_param("arm1_workspace_min")
        self.arm1_ws_max = rospy.get_param("arm1_workspace_max")
        self.arm2_ws_min = rospy.get_param("arm2_workspace_min")
        self.arm2_ws_max = rospy.get_param("arm2_workspace_max")

        # Leap motion workspace boundaries
        self.l_hand_ws_min = rospy.get_param("l_hand_workspace_min")
        self.l_hand_ws_max = rospy.get_param("l_hand_workspace_max")
        self.r_hand_ws_min = rospy.get_param("r_hand_workspace_min")
        self.r_hand_ws_max = rospy.get_param("r_hand_workspace_max")

        # Gripper setup
        if self.control_gripper_left:
            gripper1_joint_names = rospy.get_param(self.robot1_name + "/gripper_joint_names", [])
            gripper1_joint_name = gripper1_joint_names[0]
            self.gripper1_joint = self.moveit_robot.get_joint(gripper1_joint_name)
            rospy.loginfo("Left gripper control enabled")
        if self.control_gripper_right:
            gripper2_joint_names = rospy.get_param(self.robot2_name + "/gripper_joint_names", [])
            gripper2_joint_name = gripper2_joint_names[0]
            self.gripper2_joint = self.moveit_robot.get_joint(gripper2_joint_name)
            rospy.loginfo("Rigth gripper control enabled")

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

        # Subscribe to the Leap data
        self.leap_ns = rospy.get_param('leap_ns', "/leapmotion")
        right_hand_data = self.leap_ns + '/hands/right'
        left_hand_data = self.leap_ns + '/hands/left'
        self.sub_right_hand_pose = rospy.Subscriber(right_hand_data, Hand, self.right_hand_pose_callback)
        self.sub_right_hand_pinch = rospy.Subscriber(right_hand_data + '/pinch', Range, self.right_hand_pinch_callback)
        self.sub_left_hand_pose = rospy.Subscriber(left_hand_data, Hand, self.left_hand_pose_callback)
        self.sub_left_hand_pinch = rospy.Subscriber(left_hand_data + '/pinch', Range, self.left_hand_pinch_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + right_hand_data + " and " + left_hand_data)

        # Robot 1
        # Subscribe to the arm and gripper state
        self.sub_arm1_state = rospy.Subscriber('/' + self.robot1_name + '/base_feedback', BaseCyclic_Feedback, self.arm1_state_callback)
        self.sub_gripper1_state = rospy.Subscriber('/' + self.robot1_name + '/base_feedback/joint_state', JointState, self.gripper1_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot1_name + "/base_feedback")
        # Create publisher for cartesian velocity commands
        self.pub_arm1_vel = rospy.Publisher('/' + self.robot1_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)        
        if self.control_gripper_left:
            # Create publisher for gripper commands
            self.pub_gripper1 = rospy.Publisher('/' + self.robot1_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)
        # Initialize the hand pose and arm state
        self.last_left_hand_pose= None
        self.last_left_pinch_state = 0.0
        self.last_gripper1_state = None
        self.last_arm1_state = None

        # Robot 2
        # Subscribe to the arm and gripper state
        self.sub_arm2_state = rospy.Subscriber('/' + self.robot2_name + '/base_feedback', BaseCyclic_Feedback, self.arm2_state_callback)
        self.sub_gripper2_state = rospy.Subscriber('/' + self.robot2_name + '/base_feedback/joint_state', JointState, self.gripper2_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot2_name + "/base_feedback")
        # Create publisher for cartesian velocity commands
        self.pub_arm2_vel = rospy.Publisher('/' + self.robot2_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        if self.control_gripper_right:
            # Create publisher for gripper commands
            self.pub_gripper2 = rospy.Publisher('/' + self.robot2_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)
        # Initialize the hand pose and arm state
        self.last_right_hand_pose= None
        self.last_right_pinch_state = 0.0
        self.last_gripper2_state = None
        self.last_arm2_state = None

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
        max_acc = self.moveit_max_acc
        max_vel = self.moveit_max_vel
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

        # Stop the arms
        rospy.loginfo('Stopping the arm and shutting down...')
        self.stop(self.pub_arm1_vel)
        self.stop(self.pub_arm2_vel)
        
    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm states to be received
        while self.last_arm1_state is None or self.last_arm2_state is None:
            rospy.loginfo_throttle(10, "Waiting for arm state...")
            self.rate.sleep()
        # Go home
        if not self.go_to(self.ready_pose):
            rospy.logerr("Failed to go home")
            return
        rospy.loginfo("Ready to teleoperate the arms!")
        while not rospy.is_shutdown():
            # Get the current hand positions and orientations
            right_hand_pose = self.last_right_hand_pose
            left_hand_pose = self.last_left_hand_pose
            right_pinch_state = self.last_right_pinch_state
            left_pinch_state = self.last_left_pinch_state

            # Check if at least one of the hand poses is received
            if right_hand_pose is None and left_hand_pose is None:
                rospy.loginfo_throttle(10, "Waiting for hand pose...")
                if self.last_command_time != None and self.inactivity_timeout > 0 and (rospy.Time.now() - self.last_command_time).to_sec() > self.inactivity_timeout:
                    rospy.loginfo("No hand pose received in " + str(self.inactivity_timeout) + " seconds. Going to ready pose...")
                    # Go to "ready" pose
                    self.go_to(self.ready_pose)
                    self.last_command_time = None
                else:
                    self.stop(self.pub_arm1_vel)
                    self.stop(self.pub_arm2_vel)
                self.rate.sleep()
                continue

            # Get the current arm and gripper states
            arm1_state = self.last_arm1_state
            gripper1_state = self.last_gripper1_state
            arm2_state = self.last_arm2_state
            gripper2_state = self.last_gripper2_state

            if left_hand_pose is not None:
                # Transform the hand position and orientation to the base frame
                left_hand_pose = lc.leap_to_arm_workspace(left_hand_pose, self.arm1_ws_min, self.arm1_ws_max, self.l_hand_ws_min, self.l_hand_ws_max)
                # Check that the hand pose is aligned with the current arm state
                diff_left = lc.mismatch_hand_arm(left_hand_pose, arm1_state)
                if diff_left > self.mismatch_threshold:
                    rospy.loginfo_throttle(1, "Waiting for the left hand pose to align with the arms...")
                    self.stop(self.pub_arm1_vel)
                    self.last_left_hand_pose = None
                else:
                    # All good to send velocity commands
                    rospy.loginfo_throttle(60, "Sending velocity commands to the left arm...")
                    lc.reach_arm_position(left_hand_pose, arm1_state, self.pub_arm1_vel, 
                                          self.vel_scale_factor, self.max_linear_vel, self.max_angular_vel)

            if right_hand_pose is not None:
                # Transform the hand position and orientation to the base frame
                right_hand_pose = lc.leap_to_arm_workspace(right_hand_pose, self.arm2_ws_min, self.arm2_ws_max, self.r_hand_ws_min, self.r_hand_ws_max)
                # Check that the hand pose is aligned with the current arm state
                diff_right = lc.mismatch_hand_arm(right_hand_pose, arm2_state)
                if diff_right > self.mismatch_threshold:
                    rospy.loginfo_throttle(1, "Waiting for the right hand pose to align with the arms...")
                    self.stop(self.pub_arm2_vel)
                    self.last_right_hand_pose = None
                else:
                    # All good to send velocity commands
                    rospy.loginfo_throttle(60, "Sending velocity commands to the right arm...")
                    lc.reach_arm_position(right_hand_pose, arm2_state, self.pub_arm2_vel, 
                                          self.vel_scale_factor, self.max_linear_vel, self.max_angular_vel)

            # Control the grippers
            if self.control_gripper_left:
                lc.gripper_binary_control(left_pinch_state, gripper1_state, self.gripper1_joint, self.pinch_threshold)
            if self.control_gripper_right:
                lc.gripper_binary_control(right_pinch_state, gripper2_state, self.gripper2_joint, self.pinch_threshold)

            # Clear the last hand pose
            self.last_right_hand_pose = None
            self.last_left_hand_pose = None
            self.last_command_time = rospy.Time.now()

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def left_hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_left_hand_pose = data.palm_center

    
    def right_hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_right_hand_pose = data.palm_center
    

    def left_hand_pinch_callback(self, data):
        self.last_left_pinch_state = self.update_pinch_state(self.last_left_pinch_state, data.range)
    

    def right_hand_pinch_callback(self, data):
        self.last_right_pinch_state = self.update_pinch_state(self.last_right_pinch_state, data.range)
    

    def update_pinch_state(self, last_pinch_state, new_range):
        # Update the pinch state using a windowed average
        return 0.95 * last_pinch_state + 0.05 * new_range


    def arm1_state_callback(self, data):
        self.last_arm1_state = lc.process_arm_state(data, self.arm1_base_link, self.reference_frame, self.tf_buffer)


    def gripper1_state_callback(self, data):
        self.last_gripper1_state = lc.process_gripper_state(data)


    def arm2_state_callback(self, data):
        self.last_arm2_state = lc.process_arm_state(data, self.arm2_base_link, self.reference_frame, self.tf_buffer)


    def gripper2_state_callback(self, data):
        self.last_gripper2_state = lc.process_gripper_state(data)


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


    def go_to(self, pose_name, open_gripper=True):
        '''Go to the home position'''
        reach_pose = lc.reach_named_position(self.moveit_arm_group, pose_name)

        if open_gripper:
            open1 = open2 = True
            if self.control_gripper_left:
                open1 = lc.reach_gripper_position(0.0, self.gripper1_joint)
            if self.control_gripper_right:
                open2 = lc.reach_gripper_position(0.0, self.gripper2_joint)
        return reach_pose and open1 and open2

    
    def stop(self, publisher):
        arm_vel = TwistCommand()
        arm_vel.reference_frame = 0
        arm_vel.twist.linear_x = 0
        arm_vel.twist.linear_y = 0
        arm_vel.twist.linear_z = 0
        arm_vel.twist.angular_x = 0
        arm_vel.twist.angular_y = 0
        arm_vel.twist.angular_z = 0
        publisher.publish(arm_vel)


if __name__ == '__main__':
    teleop = KinovaTeleopDual()
    teleop.run()
