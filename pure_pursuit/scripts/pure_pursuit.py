#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Twist 
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import csv
import numpy as np
from numpy import linalg as LA
import math
import tf

class PurePursuit:
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        # TODO: create ROS subscribers and publishers.
        self.L = 1.0
        self.angle = 0
        self.velo = 0
        self.min_angle = -0.4
        self.max_angle = 0.4
        self.servo_offset = 0.0
        self.start = 0

        ## Driving msg
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.header.frame_id = "laser"
        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.speed = 0.0

        drive_topic = '/drive'
        teleop_topic = '/turtle1/cmd_vel'
        pf_pose_topic = 'pf/viz/inferred_pose'

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped) # Publish to drive
        self.teleop_sub = rospy.Subscriber(teleop_topic, Twist, self.teleop_callback) # Subscribe to /keyboard from teleop_key
        self.pose_sub = rospy.Subscriber(pf_pose_topic, PoseStamped, self.pose_callback)

        self.x_list = []
        self.y_list = []

        input_file = rospy.get_param("~wp_file")
        # input_file = "/home/vietanhle/data_file/wp-selected-50.csv"

        with open(input_file) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                self.x_list.append(float(line[0])) 
                self.y_list.append(float(line[1]))

        self.wp_len = len(self.x_list)


    def teleop_callback(self, msg):
        if msg.linear.x > 0:    
            self.start = 1 # push up button to start, down, right, left to stop  
        else:
            self.start = 0 

    def limitAngle(self, angle):
        # TODO: Limit / Saturate the steering angle to the range [min_angle, max_angle]
        angle = angle + self.servo_offset
        if angle > self.max_angle:
            angle = self.max_angle
        elif angle < self.min_angle:
            angle = self.min_angle
        return angle

    def selectVel(self, angle):
        # TODO: Select velocity from discrete incremental rules
        angle_abs = abs(angle)
        if angle_abs > math.radians(20):
            velocity = 0.5
        elif angle_abs > math.radians(10):
            velocity = 1.0
        else:
            velocity = 2.0
        return velocity

    def find_closest(self, current_x, current_y):
        dis_list = []
        for i in range(self.wp_len):
            distance = LA.norm(np.array([current_x-self.x_list[i], current_y-self.y_list[i]]),2)
            dis_list.append(distance)
        
        idx = np.argmin(dis_list)
        return idx

    def pose_callback(self, msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        if self.start == 1:
            current_x = msg.pose.position.x
            current_y = msg.pose.position.y

            cur_idx = self.find_closest(current_x, current_y)
            cur_L = LA.norm(np.array([current_x-self.x_list[cur_idx], current_y-self.y_list[cur_idx]]),2)
            min_L =  abs(cur_L - self.L)

            while 1:
                if cur_idx == self.wp_len - 1:
                    cur_L = LA.norm(np.array([current_x-self.x_list[0], current_y-self.y_list[0]]),2)
                else:
                    cur_L = LA.norm(np.array([current_x-self.x_list[cur_idx+1], current_y-self.y_list[cur_idx+1]]),2)
                if (abs(cur_L-self.L) < min_L):
                    min_L = abs(cur_L-self.L)
                    if cur_idx == self.wp_len - 1:
                        cur_idx = 0
                    else:
                        cur_idx += 1 
                else:
                    break 
            
            # TODO: transform goal point to vehicle frame of reference
            quaternion = np.array([msg.pose.orientation.x, 
                            msg.pose.orientation.y, 
                            msg.pose.orientation.z, 
                            msg.pose.orientation.w])
            euler = tf.transformations.euler_from_quaternion(quaternion)
            current_theta = euler[2]

            a = self.x_list[cur_idx] - current_x    
            b = self.y_list[cur_idx] - current_y

            x_r = math.cos(current_theta)*a + math.sin(current_theta)*b
            y_r = -math.sin(current_theta)*a + math.cos(current_theta)*b

            r = cur_L**2/2/abs(y_r)
            if y_r < 0:
                self.angle = self.limitAngle(-1/r)
            else:
                self.angle = self.limitAngle(1/r)

            self.velo = self.selectVel(self.angle)
            # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians
            self.drive_msg.header.stamp = rospy.Time.now()
            self.drive_msg.drive.steering_angle = self.angle 
            self.drive_msg.drive.speed = self.velo
            # rospy.loginfo("steer %.3f - speed %1.3f", self.drive_msg.drive.steering_angle, self.drive_msg.drive.speed)
        else:
            self.drive_msg.header.stamp = rospy.Time.now()
            self.drive_msg.drive.steering_angle = 0.0
            self.drive_msg.drive.speed = 0.0 

def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        pp.drive_pub.publish(pp.drive_msg)
        rate.sleep()

if __name__ == '__main__':
    main()