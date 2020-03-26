#!/usr/bin/env python

# KEYBOARD CONTROL RACECAR using turtle_teleop_key

import rospy
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

class KeyBoard:
    def __init__(self):
	    #TODO: initiate AckermannDriveStamped message that will be published
        self.ack_msg = AckermannDriveStamped()
        self.ack_msg.header.stamp = rospy.Time.now()
        self.ack_msg.header.frame_id = '001'
        self.ack_msg.drive.steering_angle = 0.0
        self.ack_msg.drive.speed = 0.0
        self.pub = rospy.Publisher('/drive', AckermannDriveStamped) # Publish to /vesc/high_level/ackermann_cmd_mux/input/nav_0
        self.sub = rospy.Subscriber('/turtle1/cmd_vel', Twist, self.callback) # Subscribe to /keyboard from teleop_key
    
    def callback(self, msg):
	    # TODO:
	    # - read Twist msg to detect direction command (up, down, left, right)
	    # - then set speed and velocity properly in ack_msg
        # - also store trigger time
        self.ack_msg.header.stamp = rospy.Time.now() # Trigger time
        if msg.linear.x < 0 and msg.angular.z == 0: # KEY_DOWN
            self.ack_msg.drive.steering_angle = 0.0 
            self.ack_msg.drive.speed = -2.0

        elif msg.linear.x > 0 and msg.angular.z == 0: # KEY_UP
            self.ack_msg.drive.steering_angle = 0.0 
            self.ack_msg.drive.speed = 2.0
        
        elif msg.linear.x == 0 and msg.angular.z > 0: # KEY_LEFT
            self.ack_msg.drive.steering_angle = 0.3 
            self.ack_msg.drive.speed = 2.0

        elif msg.linear.x == 0 and msg.angular.z < 0: # KEY_RIGHT
            self.ack_msg.drive.steering_angle = -0.3 
            self.ack_msg.drive.speed = 2.0

        else: # otherwise stop the car
            self.ack_msg.drive.steering_angle = 0.0 
            self.ack_msg.drive.speed = 0.0
        
    def process_keyboard(self):
	    # TODO: calculate `timeout` - the duration between current time (now) and
        # the last keypress (the last trigger time of callback() above stored in ack_msg).
        # If it is longer than a threshold (0.1 sec), set speed and direction to 0 to stop the car
        timeout = rospy.Time.now() - self.ack_msg.header.stamp
        if timeout.to_sec() > 0.1: # latest keypress is older than 0.1s ago
            # Set ack_msg to stop the car
            self.ack_msg.drive.steering_angle = 0.0 
            self.ack_msg.drive.speed = 0.0

        rospy.loginfo("ang: %g, vel: %g", self.ack_msg.drive.steering_angle, self.ack_msg.drive.speed)
        self.pub.publish(self.ack_msg)
        # TODO: Publish the drive command (ack_msg)


if __name__=='__main__':
    rospy.init_node("keyboard_node")
    kb = KeyBoard()
    rospy.loginfo("Initialized: keyboard_node")
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        kb.process_keyboard()
        rate.sleep()
