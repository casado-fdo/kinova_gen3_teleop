#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range
from std_msgs.msg import Empty
from kortex_driver.msg import *
from kortex_driver.srv import *
import moveit_commander
import moveit_msgs.msg
import numpy as np
import sys

DEFAULT_ROBOT_NAME = "my_gen3"
DEFAULT_ARM_GROUP_NAME = "arm"
DEFAULT_HAND =  "/leapmotion/hands/right"
ROBOT_MAX_ACCELERATION = 0.2
ROBOT_MAX_VELOCITY = 0.7

'''
Tool pose at home
tool_pose_x: 0.57802
tool_pose_y: 0.00705
tool_pose_z: 0.42327

Hand pose at home
pose_x: 0.0
pose_y: 0.0
pose_z: 0.35 (max is around 0.7)
'''

class KinovaTeleop(object):
    def __init__(self):
        self.robot_name = rospy.get_param('~robot_name', DEFAULT_ROBOT_NAME)
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)

        # Create the MoveItInterface necessary objects
        arm_group_name = rospy.get_param(rospy.get_namespace() + "arm_group_name", DEFAULT_ARM_GROUP_NAME)
        moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_arm_group = moveit_commander.MoveGroupCommander(arm_group_name)
        self.moveit_display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)
        # Limit velocity and acceleration
        self.moveit_arm_group.set_max_velocity_scaling_factor(ROBOT_MAX_VELOCITY)
        self.moveit_arm_group.set_max_acceleration_scaling_factor(ROBOT_MAX_ACCELERATION)

        self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
        if self.is_gripper_present:
            gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
            self.gripper_joint_name = gripper_joint_names[0]
        else:
            self.gripper_joint_name = ""
        self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())


        # Subscribe to the Leap data
        hand = rospy.get_param("/hand", DEFAULT_HAND)
        self.sub_hand_pose = rospy.Subscriber(hand + '/pose', PoseStamped, self.hand_pose_callback)
        self.sub_hand_grab = rospy.Subscriber(hand + '/grab', Range, self.hand_grab_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + hand) 

        # Subscribe to the arm and gripper state
        self.sub_arm_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback', BaseCyclic_Feedback, self.arm_state_callback)
        #self.sub_gripper_state = rospy.Subscriber('/' + self.robot_name + '/gripper_state', Range, self.gripper_state_callback)

        self.smooth_stop = rospy.Publisher('/' + self.robot_name + '/in/stop', Empty, queue_size=10)
        self.emergency_stop = rospy.Publisher('/' + self.robot_name + '/in/emergency_stop', Empty, queue_size=10)
        self.clear_faults = rospy.Publisher('/' + self.robot_name + '/in/clear_faults', Empty, queue_size=10)


        # Check if the robot is simulated in Gazebo
        self.is_simulation = False #rospy.get_param('~gazebo', False)
        # Create publisher for cartesian velocity commands
        if not self.is_simulation:
            self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        else:
            self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '//base/play_cartesian_trajectory', PlayCartesianTrajectoryGoal, queue_size=10)

        # Create publisher for gripper commands
        #self.pub_gripper = rospy.Publisher('/' + self.robot_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal ', GripperCommandActionGoal, queue_size=10)

        # Initialize the hand pose and arm state
        self.last_hand_pose= None
        self.last_grab_state = None
        self.last_arm_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(10)

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)
    

    def signal_term_handler(self):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.loginfo('User Keyboard interrupt. Going home...')
        # Try to go home
        if self.go_home():
            rospy.loginfo('Went home successfully')
        else:
            rospy.logerr('Failed to go home')
        # Stop the arm
        rospy.loginfo('Stopping the arm and shutting down...')
        self.stop()

    def leap_to_arm_workspace(self, hand_pose):
        '''Transform the hand position and orientation to the robot workspace'''
        hand_pose.position.x = hand_pose.position.x + 0.57802
        hand_pose.position.y = hand_pose.position.y + 0.00705
        hand_pose.position.z = (hand_pose.position.z - 0.35) + 0.42327

        return hand_pose
    
    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm state to be received
        while self.last_arm_state is None:
            rospy.loginfo_throttle(10, "Waiting for arm state...")
            self.rate.sleep()
        # Go home
        if not self.go_home():
            rospy.logerr("Failed to go home")
            return
        rospy.loginfo("Ready to teleoperate the arm!")
        while not rospy.is_shutdown():
            # Get the current hand position and orientation
            hand_pose = self.last_hand_pose
            grab_state = self.last_grab_state

            if hand_pose is None or grab_state is None:
                rospy.loginfo_throttle(10, "Waiting for hand pose...")
                self.stop()
                self.rate.sleep()
                continue

            rospy.loginfo_throttle(60, "Sending velocity commands to the arm...")

            # Get the current arm state
            arm_state = self.last_arm_state

            # Get the current gripper state
            #gripper_state = self.last_gripper_state

            # Transform the hand position and orientation to the base frame
            hand_pose = self.leap_to_arm_workspace(hand_pose)


            # Calculate the velocity command to move the arm to the hand position
            arm_vel = TwistCommand()
            arm_vel.reference_frame = 0
            arm_vel.twist.linear_x = (hand_pose.position.x - arm_state.base.tool_pose_x) * ROBOT_MAX_VELOCITY
            arm_vel.twist.linear_y = (hand_pose.position.y - arm_state.base.tool_pose_y) * ROBOT_MAX_VELOCITY
            arm_vel.twist.linear_z = (hand_pose.position.z - arm_state.base.tool_pose_z) * ROBOT_MAX_VELOCITY
            arm_vel.twist.angular_x = 0
            arm_vel.twist.angular_y = 0
            arm_vel.twist.angular_z = 0

            # Publish the velocity command
            self.pub_arm_vel.publish(arm_vel)

            # Control the gripper
            #gripper_cmd = GripperCommandActionGoal()
            #gripper_cmd.goal.command.position = 0.0
            #gripper_cmd.goal.command.max_effort = 0.0

            # Publish the gripper command
            #self.pub_gripper.publish(gripper_cmd)

            # Clear the last hand pose
            self.last_hand_pose = None
            self.last_grab_state = None

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.pose
    

    def hand_grab_callback(self, data):
        # Get the grab strength of the hand
        self.last_grab_state = data.range


    def arm_state_callback(self, data):
        # Get the current arm state
        self.last_arm_state = data


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
        # Set the arm group
        arm_group = self.moveit_arm_group
        
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # Plan the trajectory
        success_flag, trajectory_message, planning_time, error_code = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return arm_group.execute(trajectory_message, wait=True)


    def go_home(self):
        '''Go to the home position'''
        return self.reach_named_position("home")
    
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
