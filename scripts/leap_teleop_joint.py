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
from leap_motion_controller.msg import Hand
from kortex_driver.msg import *
from kortex_driver.srv import *
from control_msgs.msg import GripperCommandActionGoal
import moveit_commander
import moveit_msgs.msg
import numpy as np
import sys
from controller_manager_msgs.srv import SwitchController
from kortex_hardware.srv import ModeService
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class KinovaTeleop(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('/my_gen3/robot_name')
        rospy.init_node('kinova_teleop_' +  self.robot_name, anonymous=True)
        rospy.loginfo("Starting Kinova Gen3 teleoperation node for " + self.robot_name)

        # Get MoveIt! parameters
        self.home_pose = rospy.get_param("/my_gen3/home_pose_name")
        self.ready_pose = rospy.get_param("/my_gen3/ready_pose_name")
        self.moveit_max_vel = rospy.get_param("/my_gen3/moveit_max_vel")
        self.moveit_max_acc = rospy.get_param("/my_gen3/moveit_max_acc")
        arm_group_name = rospy.get_param('/my_gen3/arm_group_name')

        # Get teleoperation parameters
        #self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        #self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        #self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("/my_gen3/mismatch_threshold") # m
        #self.pinch_threshold = rospy.get_param("pinch_threshold") # m [0-1]
        self.control_gripper = rospy.get_param("/my_gen3/control_gripper") # true/false

        # Create the MoveItInterface necessary objects
        #moveit_commander.roscpp_initialize(sys.argv)
        #self.moveit_robot = moveit_commander.RobotCommander()
        #self.moveit_scene = moveit_commander.PlanningSceneInterface()
        #self.moveit_arm_group = moveit_commander.MoveGroupCommander(arm_group_name)
        #self.moveit_display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
        #                                                moveit_msgs.msg.DisplayTrajectory,
        #                                                queue_size=20)
        #self.moveit_arm_group.set_max_velocity_scaling_factor(self.moveit_max_vel)
        #self.moveit_arm_group.set_max_acceleration_scaling_factor(self.moveit_max_acc)

        # Robot reference frame
        self.arm_base_link = rospy.get_param("/my_gen3/arm_base_link")
        self.arm_ref_frame = rospy.get_param("/my_gen3/arm_reference_frame")
        if self.arm_base_link != self.arm_ref_frame:
            rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        self.joint_names = rospy.get_param("joint_names", [f"joint_{i+1}" for i in range(7)])
        self.srv_joint_speeds= rospy.ServiceProxy('/my_gen3/base/send_joint_speeds_command', SendJointSpeedsCommand)
   

        # Gripper setup
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


        # Subscribe to the arm and gripper state
        self.sub_arm_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback', BaseCyclic_Feedback, self.arm_state_callback)
        self.sub_gripper_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback/joint_state', JointState, self.gripper_state_callback)
        rospy.loginfo("Subscribed to arm state from /" + self.robot_name + "/base_feedback")

        # Create publisher for cartesian velocity commands
        #self.pub_arm_vel = rospy.Publisher('/' + self.robot_name + '/in/cartesian_velocity', TwistCommand, queue_size=10)
        self.pub_arm_joint_vel = rospy.Publisher("/joint_space_compliant_controller/command", JointTrajectoryPoint, queue_size=1)

        # Create publisher for gripper commands
        #self.pub_gripper = rospy.Publisher('/' + self.robot_name + '/robotiq_2f_85_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)

        # Subscribe to the desired tool pose topic
        self.sub_desired_pose_goal = rospy.Subscriber('/leap_to_gen3/tool_pose_goal', PoseStamped, self.desired_pose_callback)
        # Subscribe to the desired joint state topic
        self.sub_joint_state_goal = rospy.Subscriber('/leap_to_gen3/ik_joint_state_goal', JointState, self.desired_joint_state_callback)

        # Initialize the hand pose and arm state
        self.last_desired_pose = None
        self.last_desired_joint_state = None
        #self.last_pinch_state = 0.0
        self.last_gripper_state = None
        self.last_arm_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(100) # 500 Hz

        # Handle user keyboard interrupt properly
        #rospy.on_shutdown(self.signal_term_handler)

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

    
    def reach_gripper_position(self, relative_position):
        '''Move the gripper to a relative position between 0 and 1'''
        try:
            pos = relative_position* (self.gripper_max_absolute_pos - self.gripper_min_absolute_pos) + self.gripper_min_absolute_pos
            val = self.gripper_joint.move(pos, wait=True)
            return val
        except:
            return False 


    def quat_multiply(self, q1, q2):
        # Multiply two quaternions (Hamilton product)
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    

    def axis_angle_to_quat(self, axis, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        axis = np.array(axis) / np.linalg.norm(axis)
        s = np.sin(angle_rad / 2.0)
        x, y, z = axis * s
        w = np.cos(angle_rad / 2.0)
        return np.array([x, y, z, w])
        
    
    def desired_pose_callback(self, data):
        self.last_desired_pose = data.pose

    def desired_joint_state_callback(self, data):
        self.last_desired_joint_state = data

    
    def run(self):
        '''Main loop to control the arm'''

        # Wait for the arm state to be received
        #while self.last_arm_state is None:
        #    rospy.loginfo_throttle(10, "Waiting for arm state...")
        #    self.rate.sleep()
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
            # Get the desired hand pose and pinch state
            desired_pose = self.last_desired_pose
            desired_joint_state = self.last_desired_joint_state

            # Check if the hand pose is received
            if desired_pose is None or desired_joint_state is None:
                rospy.loginfo_throttle(10, "Waiting for goal...")
                self.stop()
                self.rate.sleep()
                continue

            # Get the current arm and gripper states
            arm_state = self.last_arm_state
            gripper_state = self.last_gripper_state

            # Check that the hand pose is aligned with the current arm state
            #diff = self.mismatch_hand_arm(desired_pose, arm_state)
            #if diff > self.mismatch_threshold:
            #    rospy.loginfo_throttle(1, "Waiting for hand pose to align with arm...")
            #    self.rate.sleep()
            #    self.stop()
            #    self.last_desired_pose = None
            #    continue

            # All good to send velocity commands
            rospy.loginfo_throttle(5, "Sending velocity commands to the arm...")

            # Control the arm
            #self.reach_arm_position(hand_pose, arm_state)
            #speed_req = kortex_driver.srv.SendJointSpeedsCommandRequest()
            #for jidx,jname in enumerate(self.joint_names):
            #    joint_speed = JointSpeed()
            #    joint_speed.joint_identifier = jidx
            #    joint_speed.value = desired_joint_state.velocity[jidx]
            #    speed_req.input.joint_speeds.append(joint_speed)
            #self.srv_joint_speeds(speed_req)
            cmd = JointTrajectoryPoint()
            cmd.positions = desired_joint_state.position
            cmd.velocities = [0.0] * len(self.joint_names)
            cmd.effort = [0.0] * len(self.joint_names)
            #cmd.velocities[0] = desired_joint_state.position[0] * 0.5
            #rospy.loginfo("Sending joint velocity command: " + str(cmd.positions))
            self.pub_arm_joint_vel.publish(cmd)

            # Control the gripper
            if self.control_gripper and self.is_gripper_present:
                if pinch_state <= self.pinch_threshold:
                    pinch_state = 0.0
                elif pinch_state >= (1 - self.pinch_threshold):
                    pinch_state = 1.0
                if (pinch_state == 0.0 or pinch_state == 1.0) and gripper_state != pinch_state:
                    rospy.loginfo("Changing gripper state to " + str(pinch_state))
                    self.reach_gripper_position(pinch_state)

            # Clear the last goal
            self.last_desired_pose = None
            self.last_desired_joint_state = None

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.palm_center
    

    def hand_pinch_callback(self, data):
        # Update the pinch state using a windowed average
        self.last_pinch_state = 0.95 * self.last_pinch_state + 0.05 * data.range


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

    def joint_state_callback(self, data):
        self.last_joint_state = data.position

    
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
        #speed_req = kortex_driver.srv.SendJointSpeedsCommandRequest()
        #for jidx,jname in enumerate(self.joint_names):
        #    joint_speed = JointSpeed()
        #    joint_speed.joint_identifier = jidx
        #    joint_speed.value = 0.0
        #    speed_req.input.joint_speeds.append(joint_speed)
        #self.srv_joint_speeds(speed_req)
        #return True
        cmd = JointTrajectoryPoint()
        #cmd.positions = [0.0] * len(self.joint_names)
        cmd.velocities = [0.0] * len(self.joint_names)
        self.pub_arm_joint_vel.publish(cmd)



if __name__ == '__main__':
    teleop = KinovaTeleop()
    teleop.run()
