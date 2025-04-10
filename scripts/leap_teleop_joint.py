#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from kortex_driver.msg import *
from kortex_driver.srv import *
import numpy as np
from controller_manager_msgs.srv import SwitchController
from kortex_hardware.srv import ModeService
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class KinovaTeleop(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)
        rospy.loginfo("Starting Kinova Gen3 teleoperation node for " + self.robot_name)

        # Get home / ready configurations
        self.home_pose = rospy.get_param("home_pose_name")
        self.ready_pose = rospy.get_param("ready_pose_name")

        # Get teleoperation parameters
        #self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        #self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        #self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m
        #self.pinch_threshold = rospy.get_param("pinch_threshold") # m [0-1]
        self.control_gripper = rospy.get_param("control_gripper") # true/false

        # Robot reference frame
        self.arm_base_link = rospy.get_param("arm_base_link")
        self.arm_ref_frame = rospy.get_param("arm_reference_frame")
        if self.arm_base_link != self.arm_ref_frame:
            rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        self.joint_names = rospy.get_param("joint_names", [f"joint_{i+1}" for i in range(7)])
   
        # Gripper setup
        if self.control_gripper:
            self.gripper_joint = None # TODO
            rospy.loginfo("Gripper control enabled")
        else:
            self.gripper_joint_name = ""
            rospy.loginfo("Gripper control disabled")
        self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())


        # Subscribe to the robot joint state, tool pose, and gripper state
        self.sub_joint_state = rospy.Subscriber('/joint_states', JointState, self.current_joint_state_callback)
        self.sub_tool_pose = rospy.Subscriber('/tool_pose', PoseStamped, self.current_tool_pose_callback)
        rospy.loginfo("Subscribed to arm state from " + self.robot_name)
        self.last_joint_state = None
        self.last_tool_pose = None
        self.last_gripper_state = None

        # Create publisher for joint space commands
        self.pub_arm_joint_vel = rospy.Publisher("/joint_space_compliant_controller/command", JointTrajectoryPoint, queue_size=1)

        # Subscribe to the desired tool pose topic
        self.sub_desired_pose_goal = rospy.Subscriber('/leap_to_gen3/tool_pose_goal', PoseStamped, self.desired_pose_callback)
        # Subscribe to the desired joint state topic
        self.sub_joint_state_goal = rospy.Subscriber('/leap_to_gen3/ik_joint_state_goal', JointState, self.desired_joint_state_callback)

        # Initialize the desired pose and joint state
        self.last_desired_pose = None
        self.last_desired_joint_state = None
        #self.last_pinch_state = 0.0
        
        # Setup the rate for the main loop
        self.rate = rospy.Rate(15) # Hz

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    

    def signal_term_handler(self):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.loginfo('User Keyboard interrupt. Going home...')
        # Try to go home
        is_home = False
        #while not is_home:
        #    TODO: home behaviour


        if not is_home:
            rospy.logerr("Failed to go home")
        else:
            rospy.loginfo("Went home successfully")

        # Stop the arm
        rospy.loginfo('Stopping the arm and shutting down...')
        self.stop()


    
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

    
    def desired_pose_callback(self, data):
        self.last_desired_pose = data.pose


    def desired_joint_state_callback(self, data):
        self.last_desired_joint_state = data

    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm state to be received
        while self.last_tool_pose is None or self.last_joint_state is None:
            rospy.loginfo_throttle(10, "Waiting for robot state...")
            self.rate.sleep()

        # Go home
        #if not self.go_to(self.ready_pose):
        #    rospy.logerr("Failed to go home")
        #    return
        rospy.loginfo("Ready to teleoperate the arm!")

        print("Switching to joint space controller. The robot might jerk a bit as it is switching to effort mode.")
        rospy.wait_for_service("controller_manager/switch_controller")
        rospy.wait_for_service("set_control_mode")
        switch_controller = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        mode_change = rospy.ServiceProxy("set_control_mode", ModeService)
        try:
            resp = switch_controller(
                ["joint_space_compliant_controller"],
                ["velocity_controller", "task_space_compliant_controller"],
                1,
                0,
                5,
            )
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        try:
            resp1 = mode_change("effort")
            print("Mode change response: ", resp1)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        print("Switch complete. You can now send commands to the robot.")

        while not rospy.is_shutdown():
            # Get the goal pose and pinch state
            desired_pose = self.last_desired_pose
            desired_joint_state = self.last_desired_joint_state

            # Check if the goal pose is received
            if desired_pose is None or desired_joint_state is None:
                rospy.loginfo_throttle(10, "Waiting for goal...")
                self.stop()
                self.rate.sleep()
                continue

            # Get the current arm and gripper states
            tool_pose = self.last_tool_pose
            gripper_state = self.last_gripper_state

            # Check that the hand pose is aligned with the current arm state
            diff = self.mismatch_hand_arm(desired_pose, tool_pose)
            if diff > self.mismatch_threshold:
                rospy.loginfo_throttle(1, "Waiting for hand pose to align with arm...")
                self.rate.sleep()
                self.stop()
                self.last_desired_pose = None
                continue

            # All good to send velocity commands
            rospy.loginfo_throttle(5, "Sending velocity commands to the arm...")

            # Control the arm
            cmd = JointTrajectoryPoint()
            cmd.positions = desired_joint_state.position
            cmd.velocities = [0.0] * len(self.joint_names)
            cmd.effort = [0.0] * len(self.joint_names)
            self.pub_arm_joint_vel.publish(cmd)

            # Control the gripper
            #if self.control_gripper:
            #    if pinch_state <= self.pinch_threshold:
            #        pinch_state = 0.0
            #    elif pinch_state >= (1 - self.pinch_threshold):
            #        pinch_state = 1.0
            #    if (pinch_state == 0.0 or pinch_state == 1.0) and gripper_state != pinch_state:
            #        rospy.loginfo("Changing gripper state to " + str(pinch_state))
            #        # TODO: set gripper state

            # Clear the last goal
            self.last_desired_pose = None
            self.last_desired_joint_state = None

            # Sleep for the remainder of the loop
            self.rate.sleep()


    def current_tool_pose_callback(self, data):
        # Get the current tool pose
        if self.arm_base_link == self.arm_ref_frame:
            self.last_tool_pose = data
        else:
            # Transform to the arm base frame
            self.last_tool_pose = self.tool_pose_to_ref_frame(data)


    def current_joint_state_callback(self, data):
        # Get the current joint state
        self.last_joint_state = np.array(data.position)[1:] # skip the first joint (finger joint)        

    
    def current_gripper_state_callback(self, data):
        # Get the current gripper state
        raw_state = data.position[self.degrees_of_freedom]
        self.last_gripper_state = 0.0 if raw_state < 0.1 else 1.0

    
    def stop(self):
        # Not really necessary, but just in case
        cmd = JointTrajectoryPoint()
        cmd.positions = self.last_joint_state
        cmd.velocities = [0.0] * len(self.joint_names)
        self.pub_arm_joint_vel.publish(cmd)


if __name__ == '__main__':
    teleop = KinovaTeleop()
    teleop.run()
