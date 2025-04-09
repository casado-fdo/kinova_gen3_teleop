#!/usr/bin/env python3

"""
ROS node that publishes the tool pose of the Gen3 robot arm
"""

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
import numpy as np


class ToolPosePublisher(object):
    def __init__(self):
        # Initialize the node
        self.robot_name = rospy.get_param('robot_name')
        rospy.init_node('tool_pose_pub', anonymous=True)
        rospy.loginfo("Starting the end effector pose publishing node for" + self.robot_name)

        # Robot reference frame
        self.arm_base_link = rospy.get_param("arm_base_link", "base_link")
        #self.arm_ref_frame = rospy.get_param("arm_reference_frame")
        #if self.arm_base_link != self.arm_ref_frame:
        #    rospy.loginfo("Transforming arm state to reference frame " + self.arm_ref_frame)

        self.arm_tool_link = rospy.get_param("tip_link", "tool_frame") #"tool_frame") #"end_effector_link" # "robotiq_arg2f_base_link" # "tool_frame")

        # Setup the rate for the main loop
        self.rate = rospy.Rate(200) # Hz

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize our publisher
        self.pub_current_tool_pose = rospy.Publisher('/tool_pose', PoseStamped, queue_size=1)
       

    def run(self):
        while not rospy.is_shutdown():
            #rospy.loginfo_throttle(5, "Publishing tool pose...")
            # Get current end effector position using tf
            transform = None
            while True:
                try:
                    transform = self.tf_buffer.lookup_transform(self.arm_base_link, self.arm_tool_link, rospy.Time())
                    break
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue
            # Create the PoseStamped message
            tool_pose = PoseStamped()
            tool_pose.header.stamp = rospy.Time.now()
            tool_pose.header.frame_id = self.arm_base_link
            tool_pose.pose.position = transform.transform.translation
            tool_pose.pose.orientation = transform.transform.rotation
            # Publish the tool pose
            self.pub_current_tool_pose.publish(tool_pose)

            # Sleep to maintain the loop rate   
            self.rate.sleep()


if __name__ == '__main__':
    pub = ToolPosePublisher()
    try:
        pub.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt Exception. Exiting...")
        pass