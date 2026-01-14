#!/usr/bin/env python3

"""
ROS node to control two Kinova Gen3 robot arms using end-effector pose commands.
"""

import rospy
import tf2_ros
from sensor_msgs.msg import Range, JointState
from geometry_msgs.msg import TwistStamped
from kortex_driver.msg import BaseCyclic_Feedback, ActionEvent, TwistCommand
from kortex_driver.srv import *
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
        self.teleop_source = rospy.get_param("teleop_source")  # 'leap_teleop' or 'lerobot'
        self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m
        self.inactivity_timeout = rospy.get_param("inactivity_timeout") # s (-1 for infinite)
        self.last_command_time = None
        rospy.loginfo("Using teleop source: " + self.teleop_source + ", "
            "velocity scale factor: " + str(self.vel_scale_factor) + ", "
            "max linear vel: " + str(self.max_linear_vel) + ", max angular vel: " + str(self.max_angular_vel))


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
        self.moveit_arm_group.set_num_planning_attempts(30)
        self.moveit_arm_group.set_planning_time(10.0)
        self.moveit_arm_group.set_goal_position_tolerance(0.001)
        self.moveit_arm_group.set_goal_orientation_tolerance(0.001)
        self.moveit_arm_group.set_goal_joint_tolerance(0.001)

        # Robot reference frame
        self.arm1_base_link = rospy.get_param("arm1_base_link")
        self.arm2_base_link = rospy.get_param("arm2_base_link")
        self.arm_ref_frame = rospy.get_param("arm_ref_frame")
        rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        # Robot workspace boundaries
        self.arm1_ws_min = rospy.get_param("arm1_workspace_min")
        self.arm1_ws_max = rospy.get_param("arm1_workspace_max")
        self.arm2_ws_min = rospy.get_param("arm2_workspace_min")
        self.arm2_ws_max = rospy.get_param("arm2_workspace_max")

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Robot 1
        # Subscribe to the teleop input
        self.sub_arm1_teleop = rospy.Subscriber('/' + self.robot1_name + '/' + self.teleop_source  + '/cartesian_velocity', TwistStamped, self.arm1_teleop_callback)
        # Create publisher for cartesian velocity commands
        self.pub_arm1_vel = rospy.Publisher('/' + self.robot1_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)        
        # Initialize the hand pose and arm state
        self.last_left_vel_command= None
        # Clear any faults
        clear_faults_srv_arm1 = '/' + self.robot1_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_srv_arm1)
        self.clear_faults_arm1 = rospy.ServiceProxy(clear_faults_srv_arm1, Base_ClearFaults)
        try:
            self.clear_faults_arm1()
        except rospy.ServiceException as e:
            rospy.logerr("Failed to clear faults for left arm: " + str(e))

        # Robot 2
        # Subscribe to the teleop input
        self.sub_arm2_teleop = rospy.Subscriber('/' + self.robot2_name + '/' + self.teleop_source  + '/cartesian_velocity', TwistStamped, self.arm2_teleop_callback)
        # Create publisher for cartesian velocity commands
        self.pub_arm2_vel = rospy.Publisher('/' + self.robot2_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        # Initialize the hand pose and arm state
        self.last_right_vel_command= None
        # Clear any faults
        clear_faults_srv_arm2 = '/' + self.robot2_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_srv_arm2)
        self.clear_faults_arm2 = rospy.ServiceProxy(clear_faults_srv_arm2, Base_ClearFaults)
        try:
            self.clear_faults_arm2()
        except rospy.ServiceException as e: 
            rospy.logerr("Failed to clear faults for right arm: " + str(e))

        # Setup the rate for the main loop
        self.rate = rospy.Rate(25) # 25 Hz

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)


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


        # Go home
        if not self.go_to(self.ready_pose):
            rospy.logerr("Failed to go home")
            return
        rospy.loginfo("Ready to teleoperate the arms!")
        while not rospy.is_shutdown():
            # Get the current hand positions and orientations
            right_vel_command = self.last_right_vel_command
            left_vel_command = self.last_left_vel_command

            # Check if at least one of the hand poses is received
            if self.is_twist_msg_empty(right_vel_command) and self.is_twist_msg_empty(left_vel_command):
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

            if not self.is_twist_msg_empty(left_vel_command):
                rospy.loginfo_throttle(60, "Sending velocity commands to the left arm...")
                lc.reach_arm_position(left_vel_command, self.pub_arm1_vel)

            if not self.is_twist_msg_empty(right_vel_command):
                rospy.loginfo_throttle(60, "Sending velocity commands to the right arm...")
                lc.reach_arm_position(right_vel_command, self.pub_arm2_vel)

            # Clear the last commands
            self.last_right_vel_command = None
            self.last_left_vel_command = None
            self.last_command_time = rospy.Time.now()

            # Sleep for the remainder of the loop
            self.rate.sleep()

    def is_twist_msg_empty(self, twist_msg):
        return twist_msg is None or twist_msg.header.stamp == rospy.Time(0) or (
            twist_msg.twist.linear.x == 0.0 and twist_msg.twist.linear.y == 0.0 and twist_msg.twist.linear.z == 0.0 and
            twist_msg.twist.angular.x == 0.0 and twist_msg.twist.angular.y == 0.0 and twist_msg.twist.angular.z == 0.0)

    def arm1_teleop_callback(self, data):
        # Get the position and orientation of the hand
        self.last_left_vel_command = data

    def arm2_teleop_callback(self, data):
        # Get the position and orientation of the hand
        self.last_right_vel_command = data


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


    def go_to(self, pose_name):
        '''Go to the home position'''
        return lc.reach_named_position(self.moveit_arm_group, pose_name)
    

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
