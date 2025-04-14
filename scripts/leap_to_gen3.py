#!/usr/bin/env python3

"""
ROS node to get the inverse kinematics for the hand pose provided by the Leap Motion
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range, JointState
from leap_motion_controller.msg import Hand
from kortex_driver.msg import *
from kortex_driver.srv import *
import numpy as np
from trac_ik_python.trac_ik import IK

# Maximum joint position difference between two consecutive commands
MAX_JOINT_DIFF = [0.15, 0.15, 0.15, 0.15, 0.3, 0.5, 0.7] # radians 

class Leap2Gen3(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('leap_to_gen3', anonymous=True)
        rospy.loginfo("Starting IK node for" + self.robot_name)

        # Get teleoperation parameters
        self.vel_scale_factor = rospy.get_param("vel_scale_factor")
        self.max_linear_vel = rospy.get_param("max_linear_vel") # m/s
        self.max_angular_vel = rospy.get_param("max_angular_vel") # rad/s
        self.mismatch_threshold = rospy.get_param("mismatch_threshold") # m

        # Robot reference frame
        self.arm_base_link = rospy.get_param("arm_base_link")
        self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

        #self.arm_ref_frame = rospy.get_param("arm_reference_frame")
        #if self.arm_base_link != self.arm_ref_frame:
        #    rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        self.arm_tool_link = rospy.get_param("tool_link", "tool_frame") #"tool_frame") #"end_effector_link" # "robotiq_arg2f_base_link" # "tool_frame")
        self.joint_names = rospy.get_param("joint_names", [f"joint_{i+1}" for i in range(7)])
        self.ik_timeout = rospy.get_param("ik_timeout", 0.0025) # seconds
        self.ik_eps = rospy.get_param("ik_eps", 1e-4) # tolerance for the IK solver

        self.ik_solver = IK(self.arm_base_link, self.arm_tool_link, self.ik_timeout, self.ik_eps, solve_type="Distance")
   
        #self.sub_arm_joint_state = rospy.Subscriber('/' + self.robot_name + '/base_feedback/joint_state', JointState, self.joint_state_callback)
        self.sub_arm_joint_state = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.last_joint_state = None
        self.q_hand_to_tool_frame = [0.5] * 4
        self.last_ik_speed = [0.0] * len(self.joint_names)

        # Robot workspace boundaries
        self.arm_ws_min = rospy.get_param("arm_workspace_min")
        self.arm_ws_max = rospy.get_param("arm_workspace_max")

        # Leap motion workspace boundaries
        self.leap_ws_min = rospy.get_param("leap_workspace_min")
        self.leap_ws_max = rospy.get_param("leap_workspace_max")

        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

        # Subscribe to the Leap data
        self.leap_ns = rospy.get_param('leap_ns', "/leapmotion")
        self.leap_hand = rospy.get_param("leap_hand", "right")
        hand_data = self.leap_ns + '/hands/' + self.leap_hand
        self.sub_hand_pose = rospy.Subscriber(hand_data, Hand, self.hand_pose_callback)
        self.sub_hand_pinch = rospy.Subscriber(hand_data + '/pinch', Range, self.hand_pinch_callback)
        rospy.loginfo("Subscribed to Leap Motion data from " + hand_data) 

        # Initialize the hand pose and arm state
        self.last_hand_pose= None
        self.last_pinch_state = 0.0
        self.last_arm_state = None

        # Setup the rate for the main loop
        self.rate = rospy.Rate(100)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize our publishers
        self.pub_tool_pose_goal = rospy.Publisher('/leap_to_gen3/tool_pose_goal', PoseStamped, queue_size=1)
        self.pub_ik_joint_state_goal = rospy.Publisher('/leap_to_gen3/ik_joint_state_goal', JointState, queue_size=1)


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
        
    
    def run(self):
        '''Main loop'''
        # Wait for the arm state to be received
        while self.last_joint_state is None:
            rospy.loginfo_throttle(10, "Waiting for arm state...")
            self.rate.sleep()
        
        rospy.loginfo("Ready to perform IK!")
        while not rospy.is_shutdown():
            # Get the current hand position and orientation
            hand_pose = self.last_hand_pose

            # Check if the hand pose is received
            if hand_pose is None:
                rospy.loginfo_throttle(10, "Waiting for hand pose...")
                self.rate.sleep()
                continue

            if self.last_joint_state is None:
                rospy.loginfo_throttle(10, "Waiting for arm state...")
                self.rate.sleep()
                continue

            # Transform the hand position and orientation to the base frame
            hand_pose = self.leap_to_arm_workspace(hand_pose)

            rospy.loginfo_throttle(5, "Transforming hand pose to Gen3 workspace...")

            x, y, z = hand_pose.position.x, hand_pose.position.y, hand_pose.position.z
            qx, qy, qz, qw = hand_pose.orientation.x, hand_pose.orientation.y, hand_pose.orientation.z, hand_pose.orientation.w

            # Convert orientation of the hand to the end effector frame
            q_result = self.quat_multiply([qx, qy, qz, qw], self.q_hand_to_tool_frame)
            qx, qy, qz, qw = q_result[0], q_result[1], q_result[2], q_result[3]

            time = rospy.Time.now()
            
            # Publish the tool pose goal
            tool_pose_goal = PoseStamped()
            tool_pose_goal.header.stamp = time
            tool_pose_goal.header.frame_id = self.arm_base_link
            tool_pose_goal.pose.position = hand_pose.position
            tool_pose_goal.pose.orientation.x = qx
            tool_pose_goal.pose.orientation.y = qy
            tool_pose_goal.pose.orientation.z = qz
            tool_pose_goal.pose.orientation.w = qw
            self.pub_tool_pose_goal.publish(tool_pose_goal)

            # Calculate the IK solution starting from the last arm joint state
            seed = self.last_joint_state[:self.degrees_of_freedom]
            ik_solution = self.ik_solver.get_ik(seed, x, y, z, qx, qy, qz, qw)

            if ik_solution is not None:
                # Check the difference between the desired and current joint angles
                new_position = []
                for jidx,jname in enumerate(self.joint_names):
                    # Angle difference for joints in radians
                    diff = np.arctan2(np.sin(ik_solution[jidx] - self.last_joint_state[jidx]),
                                    np.cos(ik_solution[jidx] - self.last_joint_state[jidx]))
                    # Limit the difference to a maximum value
                    diff = np.clip(diff, -MAX_JOINT_DIFF[jidx], MAX_JOINT_DIFF[jidx])    
                    # The new position is the current position plus the difference
                    new_position.append(self.last_joint_state[jidx] + diff)

                # Publish the joint state goal
                joint_state = JointState()
                joint_state.header.stamp = time
                joint_state.name = self.joint_names
                joint_state.position = new_position 
                self.pub_ik_joint_state_goal.publish(joint_state)
            
            else:
                rospy.logerr("IK solution not found")

            # Clear the last hand pose
            self.last_hand_pose = None
            self.last_joint_state = None

            # Sleep for the remainder of the loop
            self.rate.sleep()
        
    
    def hand_pose_callback(self, data):
        # Get the position and orientation of the hand
        self.last_hand_pose = data.palm_center
    

    def hand_pinch_callback(self, data):
        # Update the pinch state using a windowed average
        self.last_pinch_state = 0.95 * self.last_pinch_state + 0.05 * data.range
       

    def joint_state_callback(self, data):
        self.last_joint_state = np.array(data.position)[1:] # skip the first joint (finger joint)       


if __name__ == '__main__':
    leap2gen3 = Leap2Gen3()
    leap2gen3.run()
