#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
from sensor_msgs.msg import Range, JointState
from leap_motion_controller.msg import Hand
from kortex_driver.msg import TwistCommand, BaseCyclic_Feedback, ActionEvent
from kortex_driver.srv import *
from control_msgs.msg import GripperCommandActionGoal
import moveit_commander
import moveit_msgs.msg
import numpy as np
import sys
import leap_common as lc


class KinovaTeleopSingle(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)
        rospy.loginfo("Starting Kinova Gen3 teleoperation node for " + self.robot_name)

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
        self.control_gripper = rospy.get_param("control_gripper") # true/false
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
        self.is_gripper_present = rospy.get_param(self.robot_name + "/is_gripper_present", False)
        if self.control_gripper and self.is_gripper_present:
            gripper_joint_names = rospy.get_param(self.robot_name + "/gripper_joint_names", [])
            gripper_joint_name = gripper_joint_names[0]
            self.gripper_joint = self.moveit_robot.get_joint(gripper_joint_name)
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
        self.sub_hand_pose = rospy.Subscriber(hand_data, Hand, self.hand_pose_callback)
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

        # Stop the arm
        rospy.loginfo('Stopping the arm and shutting down...')
        self.stop()

    
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
                if self.last_command_time != None and self.inactivity_timeout > 0 and (rospy.Time.now() - self.last_command_time).to_sec() > self.inactivity_timeout:
                    rospy.loginfo("No hand pose received in " + str(self.inactivity_timeout) + " seconds. Going to ready pose...")
                    # Go to "ready" pose
                    self.go_to(self.ready_pose)
                    self.last_command_time = None
                else:
                    self.stop()
                self.rate.sleep()
                continue

            # Get the current arm and gripper states
            arm_state = self.last_arm_state
            gripper_state = self.last_gripper_state

            # Transform the hand position and orientation to the base frame
            hand_pose = lc.leap_to_arm_workspace(hand_pose, self.arm_ws_min, self.arm_ws_max, self.leap_ws_min, self.leap_ws_max)

            # Check that the hand pose is aligned with the current arm state
            diff = lc.mismatch_hand_arm(hand_pose, arm_state)
            if diff > self.mismatch_threshold:
                rospy.loginfo_throttle(1, "Waiting for hand pose to align with arm...")
                self.rate.sleep()
                self.stop()
                self.last_hand_pose = None
                continue

            # All good to send velocity commands
            rospy.loginfo_throttle(60, "Sending velocity commands to the arm...")

            # Control the arm
            lc.reach_arm_position(hand_pose, arm_state, self.pub_arm_vel,
                                  self.vel_scale_factor, self.max_linear_vel, self.max_angular_vel)

            # Control the gripper
            if self.control_gripper and self.is_gripper_present:
                lc.gripper_binary_control(pinch_state, gripper_state, self.gripper_joint, self.pinch_threshold)

            # Clear the last hand pose
            self.last_hand_pose = None
            self.last_command_time = rospy.Time.now()

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.palm_center
    

    def hand_pinch_callback(self, data):
        # Update the pinch state using a windowed average
        self.last_pinch_state = 0.95 * self.last_pinch_state + 0.05 * data.range


    def arm_state_callback(self, data):
        self.last_arm_state = lc.process_arm_state(data, self.arm_base_link, self.arm_ref_frame, self.tf_buffer)

    
    def gripper_state_callback(self, data):
        self.last_gripper_state = lc.process_gripper_state(data)


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
        if open_gripper and self.control_gripper and self.is_gripper_present:
            open = lc.reach_gripper_position(0.0, self.gripper_joint)
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
    teleop = KinovaTeleopSingle()
    teleop.run()
