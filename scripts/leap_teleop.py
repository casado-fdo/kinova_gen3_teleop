#!/usr/bin/env python3

"""
ROS node to control the Kinova Gen3 robot arm with a Robotiq 2F-85 gripper using the Leap Motion Controller.
"""

import rospy
import tf2_ros
import tf
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
MAX_LINEAR_VEL = 2.0 # m/s
MAX_ANGULAR_VEL = np.pi # rad/s
VEL_SCALE_FACTOR = 5.0
LEAP_WS_MIN = [-0.35, -0.45, 0.1] # m in leap motion frame
LEAP_WS_MAX = [0.35, 0.45, 0.7] # m in leap motion frame
ARM_WS_MIN = [0.0, -1.3, 0.0] # m in arm base frame
ARM_WS_MAX = [1.6, 1.3, 1.6] # m in arm base frame
ARM_ORIGIN_X = 0.57802
ARM_ORIGIN_Y = 0.00705
ARM_ORIGIN_Z = 0.42327
MISMATCH_THRESHOLD = 0.5


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
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)

        # Create the MoveItInterface necessary objects
        arm_group_name = rospy.get_param("~arm_group_name", "arm")
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

        # Gripper setup
        self.control_gripper = rospy.get_param("~control_gripper", False)
        self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
        if self.control_gripper and self.is_gripper_present:
            gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
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
        self.leap_ns = rospy.get_param('~leap_ns', "/leapmotion")
        self.leap_hand = rospy.get_param("~leap_hand", "right")
        hand_data = self.leap_ns + '/hands/' + self.leap_hand
        self.sub_hand_pose = rospy.Subscriber(hand_data + '/pose', PoseStamped, self.hand_pose_callback)
        self.sub_hand_grab = rospy.Subscriber(hand_data + '/grab', Range, self.hand_grab_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + hand_data) 

        # Subscribe to the arm and gripper state
        self.sub_arm_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback', BaseCyclic_Feedback, self.arm_state_callback)
        self.sub_gripper_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback/joint_state', JointState, self.gripper_state_callback)

        # Check if the robot is simulated in Gazebo
        self.is_simulation = False #rospy.get_param('~gazebo', False)
        # Create publisher for cartesian velocity commands
        if not self.is_simulation:
            self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        else:
            self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '/base/play_cartesian_trajectory', PlayCartesianTrajectoryGoal, queue_size=10)

        # Create publisher for gripper commands
        self.pub_gripper = rospy.Publisher('/' + self.robot_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)

        # Initialize the hand pose and arm state
        self.last_hand_pose= None
        self.last_grab_state = 0.0
        self.last_gripper_state = None
        self.last_arm_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(50)

        # Handle user keyboard interrupt properly
        rospy.on_shutdown(self.signal_term_handler)
    

    def signal_term_handler(self):
        '''Handles KeyboardInterrupts to ensure smooth exit'''
        rospy.loginfo('User Keyboard interrupt. Going home...')
        # Try to go home
        max_acc = MOVEIT_MAX_ACC
        max_vel = MOVEIT_MAX_VEL
        is_home = False
        while not is_home and max_acc >= 0.01:
            rospy.loginfo("Trying to go home (max_acc=" + str(max_acc) + ", max_vel=" + str(max_vel) + ")")
            is_home = self.go_home()
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
        hand_pose.position.x = (hand_pose.position.x - LEAP_WS_MIN[0]) / (LEAP_WS_MAX[0] - LEAP_WS_MIN[0])
        hand_pose.position.y = (hand_pose.position.y - LEAP_WS_MIN[1]) / (LEAP_WS_MAX[1] - LEAP_WS_MIN[1])
        hand_pose.position.z = (hand_pose.position.z - LEAP_WS_MIN[2]) / (LEAP_WS_MAX[2] - LEAP_WS_MIN[2])

        # Apply linear scaling to the arm workspace
        hand_pose.position.x = hand_pose.position.x * (ARM_WS_MAX[0] - ARM_WS_MIN[0]) + ARM_WS_MIN[0]
        hand_pose.position.y = hand_pose.position.y * (ARM_WS_MAX[1] - ARM_WS_MIN[1]) + ARM_WS_MIN[1]
        hand_pose.position.z = hand_pose.position.z * (ARM_WS_MAX[2] - ARM_WS_MIN[2]) + ARM_WS_MIN[2]

        rospy.loginfo("Hand pose: x=" + str(hand_pose.position.x) + ", y=" + str(hand_pose.position.y) + ", z=" + str(hand_pose.position.z))
        return hand_pose
    

    def mismatch_hand_arm(self, hand_pose, arm_state):
        '''Check if the hand pose is aligned with the arm state. Returns the euclidean distance between the two'''
        return np.sqrt((hand_pose.position.x - arm_state.base.tool_pose_x)**2 + (hand_pose.position.y - arm_state.base.tool_pose_y)**2 + (hand_pose.position.z - arm_state.base.tool_pose_z)**2)
    

    def scale_vel_log(self, vel):
        '''Logarithmically scale the velocity'''
        return np.sign(vel) * np.log(1 + abs(vel) * VEL_SCALE_FACTOR) / np.log(1 + VEL_SCALE_FACTOR)
    

    def reach_arm_position(self, hand_pose, arm_state):
        # Calculate the velocity command to move the arm to the hand position
        arm_vel = TwistCommand()
        arm_vel.reference_frame = 0
        # Linear velocity in m/s
        pos_x_diff = (hand_pose.position.x - arm_state.base.tool_pose_x)
        pos_y_diff = (hand_pose.position.y - arm_state.base.tool_pose_y)
        pos_z_diff = (hand_pose.position.z - arm_state.base.tool_pose_z)
        arm_vel.twist.linear_x = np.clip(self.scale_vel_log(pos_x_diff), -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        arm_vel.twist.linear_y = np.clip(self.scale_vel_log(pos_y_diff), -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        arm_vel.twist.linear_z = np.clip(self.scale_vel_log(pos_z_diff), -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        
        # Convert orientation from quaternion to euler angles
        q = [hand_pose.orientation.x, hand_pose.orientation.y, hand_pose.orientation.z, hand_pose.orientation.w]
        hand_theta_x, hand_theta_y, hand_theta_z = tf.transformations.euler_from_quaternion(q)

        # Add correction to the orientation of the hand
        hand_theta_x = - hand_theta_x 
        hand_theta_y = hand_theta_y + np.pi/2
        hand_theta_z = hand_theta_z + np.pi/2

        # Get enf effector orientation in euler
        arm_state_theta_x = np.radians(arm_state.base.tool_pose_theta_x)
        arm_state_theta_y = np.radians(arm_state.base.tool_pose_theta_y)
        arm_state_theta_z = np.radians(arm_state.base.tool_pose_theta_z)
        
        theta_x_diff = hand_theta_y - arm_state_theta_x
        theta_y_diff = hand_theta_z - arm_state_theta_z
        theta_z_diff = - (hand_theta_x - arm_state_theta_y)
        arm_vel.twist.angular_x = np.clip(self.scale_vel_log(theta_x_diff), -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        arm_vel.twist.angular_y = np.clip(self.scale_vel_log(theta_y_diff), -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        arm_vel.twist.angular_z = self.scale_vel_log(theta_z_diff) # no need to clip since this rotation is alreadu super slow

        # Publish the velocity command
        self.pub_arm_vel.publish(arm_vel)

    
    def reach_gripper_position(self, relative_position):
        try:
            val = self.gripper_joint.move(relative_position* (self.gripper_max_absolute_pos - self.gripper_min_absolute_pos) + self.gripper_min_absolute_pos, wait=True)
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
        if not self.go_home():
            rospy.logerr("Failed to go home")
            return
        rospy.loginfo("Ready to teleoperate the arm!")
        while not rospy.is_shutdown():
            # Get the current hand position and orientation
            hand_pose = self.last_hand_pose
            grab_state = self.last_grab_state

            # Check if the hand pose is received
            if hand_pose is None or grab_state is None:
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
                continue

            # All good to send velocity commands
            rospy.loginfo_throttle(60, "Sending velocity commands to the arm...")

            # Control the arm
            self.reach_arm_position(hand_pose, arm_state)

            # Control the gripper
            if self.control_gripper and self.is_gripper_present:
                grab_state = 1.0 if grab_state > 0.5 else 0.0
                gripper_state = 1.0 if gripper_state > 0.1 else 0.0
                if gripper_state != grab_state:
                    self.reach_gripper_position(grab_state)

            # Clear the last hand pose
            self.last_hand_pose = None

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.pose
    

    def hand_grab_callback(self, data):
        # Update the grab state using a windowed average
        self.last_grab_state = 0.99 * self.last_grab_state + 0.01 * data.range


    def arm_state_callback(self, data):
        # Get the current arm state
        self.last_arm_state = data

    
    def gripper_state_callback(self, data):
        # Get the current gripper state
        self.last_gripper_state = data.position[self.degrees_of_freedom]


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
        home = self.reach_named_position("home")
        if self.control_gripper and self.is_gripper_present:
            open = self.reach_gripper_position(0.0)
        else:
            open = True
        return home and open
    
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
