#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float32
from kortex_driver.msg import *
from kortex_driver.srv import *
import numpy as np
from controller_manager_msgs.srv import SwitchController
from kortex_hardware.srv import ModeService

# Maximum joint position difference between two consecutive commands
MAX_JOINT_DIFF = [0.15, 0.15, 0.15, 0.2, 0.25, 0.25, 0.25] # radians 
GOTO_TIMEOUT = 8.0 # seconds
ARM_DOFS = 7

class KinovaTeleop(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)
        rospy.loginfo("Starting Kinova Gen3 teleoperation node for " + self.robot_name)

        # Get home / ready configurations
        self.arm_home = rospy.get_param("home_arm_joint_state")
        self.arm_ready = rospy.get_param("ready_arm_joint_state")

        

        # Get teleoperation parameters
        #self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        #self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        #self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m
        #self.pinch_threshold = rospy.get_param("pinch_threshold") # m [0-1]

        # Robot reference frame
        self.arm_base_link = rospy.get_param("arm_base_link")
        self.arm_ref_frame = rospy.get_param("arm_reference_frame")
        if self.arm_base_link != self.arm_ref_frame:
            rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        self.arm_dofs = ARM_DOFS
        self.joint_names = rospy.get_param("joint_names", [f"joint_{i+1}" for i in range(self.arm_dofs)])
   
        # Gripper setup
        self.control_gripper = rospy.get_param("control_gripper") # true/false
        if self.control_gripper:
            rospy.loginfo("Gripper control enabled")
            self.gripper_home = rospy.get_param("home_gripper_position")
            self.gripper_ready = rospy.get_param("ready_gripper_position")
            self.joint_names.append("finger_joint")
        else:
            rospy.loginfo("Gripper control disabled")
            self.gripper_home = None
            self.gripper_ready = None   

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
        # Subscribe to the desired gripper state topic
        self.sub_gripper_pose_goal = rospy.Subscriber('/leap_to_gen3/gripper_pose_goal', Float32, self.current_gripper_state_callback)

        # Initialize the desired pose and joint state
        self.last_desired_pose = None
        self.last_desired_joint_state = None
        #self.last_pinch_state = 0.0
        
        # Setup the rate for the main loop
        self.rate = rospy.Rate(75) # Hz

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    

    def signal_term_handler(self):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.loginfo('User Keyboard interrupt. Going home...')
        # Try to go home
        rospy.loginfo('Going home...')
        is_home = self.go_to(self.arm_home, self.gripper_home)

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
        self.last_desired_joint_state = data.position


    def switch_to_joint_effort_mode(self):
        rospy.loginfo("Switching to joint space controller. The robot might jerk a bit as it is switching to effort mode...")
        rospy.wait_for_service("controller_manager/switch_controller")
        rospy.wait_for_service("set_control_mode")
        switch_controller = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        mode_change = rospy.ServiceProxy("set_control_mode", ModeService)
        try:
            switch_controller(
                ["joint_space_compliant_controller"],
                ["velocity_controller", "task_space_compliant_controller"],
                1,
                0,
                5,
            )
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s" % e)

        try:
            resp = mode_change("effort")
            rospy.loginfo("Mode change response: " + str(resp))
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s" % e)

        rospy.loginfo("Switch complete!")

    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm state to be received
        while self.last_tool_pose is None or self.last_joint_state is None:
            rospy.loginfo_throttle(10, "Waiting for robot state...")
            self.rate.sleep()

        # Switch to joint space controller
        self.switch_to_joint_effort_mode()

        # Go to ready position
        rospy.loginfo('Going to ready position...')
        self.go_to(self.arm_ready, self.gripper_ready)
        #    rospy.logerr("Failed to go home")
        #    return
        rospy.loginfo("Ready to teleoperate the arm!")


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

            if self.control_gripper:
                desired_joint_state = desired_joint_state + (gripper_state,)

            # Control the arm
            cmd = JointTrajectoryPoint()
            cmd.positions = desired_joint_state
            cmd.velocities = [0.0] * len(cmd.positions)
            cmd.effort = [0.0] * len(cmd.positions)
            self.pub_arm_joint_vel.publish(cmd)

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
        # Get only the arm joints
        self.last_joint_state = np.array(data.position)[1:] # skip the first joint (finger joint)
        if self.control_gripper:
            # Move the gripper joint to the end of the list
            self.last_joint_state = np.append(self.last_joint_state, data.position[0])       

    
    def current_gripper_state_callback(self, data):
        # Get the current gripper state
        self.last_gripper_state = data.data


    def go_to(self, target_joint_state, target_gripper_pos=None):
        '''Go to the target joint state in multiple steps'''
        # Check if the target joint state is valid
        if len(target_joint_state) != self.arm_dofs:
            rospy.logerr("Invalid joint state length")
            return False

        rospy.loginfo("Trying to go to target joint state: " + str(target_joint_state))
        time = rospy.get_time()
        while not rospy.is_shutdown():
            # Check if the time limit is reached
            if rospy.get_time() - time > GOTO_TIMEOUT:
                rospy.logerr("Timed out while trying to go to target joint state")
                return False
            new_joint_pos = []
            diffs = []
            for i in range(self.arm_dofs):
                # Check the difference between the current and target joint state
                diff = np.arctan2(
                    np.sin(target_joint_state[i] - self.last_joint_state[i]),
                    np.cos(target_joint_state[i] - self.last_joint_state[i]),
                )
                diffs.append(diff)
                # Limit the difference to a maximum value
                diff = np.clip(diff, -MAX_JOINT_DIFF[i], MAX_JOINT_DIFF[i])
                # The new position is the current position plus the difference
                new_joint_pos.append(self.last_joint_state[i] + diff)

            # Check if the gripper position is valid [0.0, 1.0]
            if self.control_gripper and target_gripper_pos is not None and target_gripper_pos >= 0.0 and target_gripper_pos <= 1.0:
                # Add the gripper joint to the new joint position
                new_joint_pos.append(target_gripper_pos)
            
            # Check if the new joint position is similar enough to the target joint state using diffs vector
            if np.all(np.abs(diffs) < 0.15):
                rospy.loginfo("Reached target joint state")
                return True
          
            # Publish the new joint position
            cmd = JointTrajectoryPoint()
            cmd.positions = new_joint_pos
            cmd.velocities = [0.0] * len(new_joint_pos)
            cmd.effort = [0.0] * len(new_joint_pos)
            self.pub_arm_joint_vel.publish(cmd)
            # Sleep for a short time to allow the arm to move
            self.rate.sleep()

    
    def stop(self):
        # Not really necessary, but just in case
        cmd = JointTrajectoryPoint()
        cmd.positions = self.last_joint_state
        cmd.velocities = [0.0] * len(self.joint_names)
        self.pub_arm_joint_vel.publish(cmd)


if __name__ == '__main__':
    teleop = KinovaTeleop()
    teleop.run()
