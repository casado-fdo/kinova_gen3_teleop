#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm using the Leap Motion Controller.
"""

import rospy
import tf2_ros
from sensor_msgs.msg import Range, JointState
from geometry_msgs.msg import TwistStamped
from leap_motion_controller.msg import Hand
from kortex_driver.msg import BaseCyclic_Feedback, ActionEvent
from kortex_driver.srv import *
from control_msgs.msg import GripperCommandActionGoal
import numpy as np
import sys
import leap_common as lc


class LeapToKinovaDual(object):
    def __init__(self):
        # Initialize the node
        self.robot1_name = rospy.get_param('robot1_name')
        self.robot2_name = rospy.get_param('robot2_name')
        rospy.init_node('leap_to_kinova_dual', anonymous=True)
        rospy.loginfo("Starting Mapping from Leap Motion to Kinova Gen3 teleop commands")

        # Get MoveIt! parameters
        self.home_pose = rospy.get_param("home_pose_name")
        self.ready_pose = rospy.get_param("ready_pose_name")
        arm_group_name = rospy.get_param('arm_group_name')

        # Get teleoperation parameters
        self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m
        self.inactivity_timeout = rospy.get_param("inactivity_timeout") # s (-1 for infinite)
        self.last_command_time = None

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

        # Leap motion workspace boundaries
        self.l_hand_ws_min = rospy.get_param("l_hand_workspace_min")
        self.l_hand_ws_max = rospy.get_param("l_hand_workspace_max")
        self.r_hand_ws_min = rospy.get_param("r_hand_workspace_min")
        self.r_hand_ws_max = rospy.get_param("r_hand_workspace_max")

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

        # Subscribe to the Leap data
        self.leap_ns = rospy.get_param('leap_ns', "/leapmotion")
        right_hand_data = self.leap_ns + '/hands/right'
        left_hand_data = self.leap_ns + '/hands/left'
        self.sub_right_hand_pose = rospy.Subscriber(right_hand_data, Hand, self.right_hand_pose_callback)
        self.sub_left_hand_pose = rospy.Subscriber(left_hand_data, Hand, self.left_hand_pose_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + right_hand_data + " and " + left_hand_data)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Robot 1
        # Subscribe to the arm
        self.sub_arm1_state = rospy.Subscriber('/' + self.robot1_name + '/base_feedback', BaseCyclic_Feedback, self.arm1_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot1_name + "/base_feedback")
        # Create publisher for cartesian velocity commands
        self.pub_arm1_vel = rospy.Publisher('/' + self.robot1_name + '/leap_teleop/cartesian_velocity', TwistStamped, queue_size=10)
        # Initialize the hand pose and arm state
        self.last_left_hand_pose= None
        self.last_arm1_state = None
       
        # Robot 2
        # Subscribe to the arm
        self.sub_arm2_state = rospy.Subscriber('/' + self.robot2_name + '/base_feedback', BaseCyclic_Feedback, self.arm2_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot2_name + "/base_feedback")
        # Create publisher for cartesian velocity commands
        self.pub_arm2_vel = rospy.Publisher('/' + self.robot2_name + '/leap_teleop/cartesian_velocity', TwistStamped, queue_size=10)
        # Initialize the hand pose and arm state
        self.last_right_hand_pose= None
        self.last_arm2_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(30)  # 30 Hz


    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm states to be received
        while self.last_arm1_state is None or self.last_arm2_state is None:
            rospy.loginfo_throttle(10, "Waiting for arm state...")
            self.rate.sleep()
        
        rospy.loginfo("Ready to map Leap Motion commands into Kinova commands!")
        while not rospy.is_shutdown():
            # Get the current hand positions and orientations
            right_hand_pose = self.last_right_hand_pose
            left_hand_pose = self.last_left_hand_pose
            
            # Check if at least one of the hand poses is received
            if right_hand_pose is None and left_hand_pose is None:
                rospy.loginfo_throttle(10, "Waiting for hand pose...")
                self.rate.sleep()
                continue

            # Get the current arm states
            arm1_state = self.last_arm1_state
            arm2_state = self.last_arm2_state

            if left_hand_pose is not None:
                # Transform the hand position and orientation to the base frame
                left_hand_pose = lc.leap_to_arm_workspace(left_hand_pose, self.arm1_ws_min, self.arm1_ws_max, self.l_hand_ws_min, self.l_hand_ws_max)            
                # Check that the hand pose is aligned with the current arm state
                diff_left = lc.mismatch_hand_arm(left_hand_pose, arm1_state)
                if diff_left > self.mismatch_threshold:
                    rospy.loginfo_throttle(1, "Waiting for the left hand pose to align with the arms...")
                    self.last_left_hand_pose = None
                else:
                    # All good to send velocity commands
                    lc.hand_pose_to_kinova_goal(left_hand_pose, arm1_state, self.pub_arm1_vel,
                                        self.arm1_base_link, self.arm_ref_frame, self.tf_buffer, 
                                        self.vel_scale_factor, self.max_linear_vel, self.max_angular_vel)

            if right_hand_pose is not None:
                # Transform the hand position and orientation to the base frame
                right_hand_pose = lc.leap_to_arm_workspace(right_hand_pose, self.arm2_ws_min, self.arm2_ws_max, self.r_hand_ws_min, self.r_hand_ws_max)
                # Check that the hand pose is aligned with the current arm state
                diff_right = lc.mismatch_hand_arm(right_hand_pose, arm2_state)
                if diff_right > self.mismatch_threshold:
                    rospy.loginfo_throttle(1, "Waiting for the right hand pose to align with the arms...")
                    self.last_right_hand_pose = None
                else:
                    # All good to send velocity commands
                    lc.hand_pose_to_kinova_goal(right_hand_pose, arm2_state, self.pub_arm2_vel,
                                        self.arm2_base_link, self.arm_ref_frame, self.tf_buffer, 
                                        self.vel_scale_factor, self.max_linear_vel, self.max_angular_vel)

            # Clear the last hand pose
            self.prev_left_hand_pose = self.last_left_hand_pose
            self.prev_right_hand_pose = self.last_right_hand_pose
            self.last_right_hand_pose = None
            self.last_left_hand_pose = None
            self.last_command_time = rospy.Time.now()

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def left_hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_left_hand_pose = data.palm_center_filtered

    
    def right_hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_right_hand_pose = data.palm_center_filtered


    def arm1_state_callback(self, data):
        self.last_arm1_state = lc.process_arm_state(data, self.arm1_base_link, self.arm_ref_frame, self.tf_buffer)


    def arm2_state_callback(self, data):
        self.last_arm2_state = lc.process_arm_state(data, self.arm2_base_link, self.arm_ref_frame, self.tf_buffer)


if __name__ == '__main__':
    teleop = LeapToKinovaDual()
    teleop.run()
