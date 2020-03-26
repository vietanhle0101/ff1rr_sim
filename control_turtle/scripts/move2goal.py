#!/usr/bin/env python
import rospy

# Import message types
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_msgs.msg import Empty  # Empty message for notification of reaching goal

# Some math functions for the computation
from math import atan2, sqrt, pi

def normalize_angle(theta):
    """Normalize an angle in randian to between -pi and pi."""
    q, r = divmod(theta, 2*pi)  # r is between 0 and 2*pi
    return ((r - 2*pi) if r > pi else r)

def saturation(v, vmin, vmax):
	"""Saturate a value between vmax and vmin."""
	return min(vmax, max(vmin, v))

class MoveToGoal:
    def __init__(self):
        # Creates a node with name 'turtle_controller'
        rospy.init_node('turtle_controller')
        #  Publisher for publishing control commands to 'turtle1/cmd_vel'.
        self.vel_pub = rospy.Publisher('turtle1/cmd_vel', Twist)
        # A subscriber to the topic 'turtle1/pose'.
        # self.update_pose is called to store the latest pose
        self.pose_sub = rospy.Subscriber('turtle1/pose', Pose, self.update_pose)
        # A subscriber to the topic 'turtle1ctrl/goal'.
        # self.update_goal is called to store the goal
        self.goal_sub = rospy.Subscriber('turtle1ctrl/goal', Pose, self.update_goal)
        # A publisher for notification of reaching the current goal
        # to the topic 'turtle1ctrl/reach'
        self.reach_pub = rospy.Publisher('turtle1ctrl/reach', Empty)

        self.pose = Pose()      # The latest pose
        self.goal_pose = Pose() # The current goal pose
        self.distance_tolerance = 0.01  # Tolerance of distance to goal

        # Controllers' parameters
        self.Kp_lin = 1.5
        self.Kp_ang = 6.0

        # Limits of velocities
        self.lin_vel_max = 2.
        self.lin_vel_min = 0.
        self.ang_vel_max = 2*pi
        self.ang_vel_min = -2*pi

        self.rate = rospy.Rate(10)      # The control rate in Hertz

        self.started = False    # Only start controlling after the first goal is received
        self.reached_goal = False    # Whether the goal has been reached
                                     # so that only one notification is sent

    def update_pose(self, data):
        """Callback function which is called when a new message of type Pose is
        received by the subscriber.  It saves the latest pose."""
        self.pose = data
        self.pose.x = round(self.pose.x, 4)     # Rounded to avoid computation noise
        self.pose.y = round(self.pose.y, 4)     # Rounded to avoid computation noise

    def update_goal(self, data):
        """Callback function to save the latest goal."""
        self.goal_pose = data
        self.started = True
        self.reached_goal = False
        rospy.loginfo("Goal set: x = %g, y = %g", data.x, data.y)

    def euclidean_distance(self):
        """Euclidean distance between current pose and the goal."""
        return sqrt((self.pose.x - self.goal_pose.x)**2 + (self.pose.y - self.goal_pose.y)**2)

    def linear_vel(self):
	    # TODO: calculate and return the linear velocity (Kp_lin * distance to goal),
	    # saturated between lin_vel_min and lin_vel_max
        v_lin = self.Kp_lin * self.euclidean_distance()
        return saturation(v_lin, self.lin_vel_min, self.lin_vel_max)

    def steering_angle(self):
	    # TODO: calculate and return the angle between the current position and the goal
        return atan2(self.goal_pose.y - self.pose.y, self.goal_pose.x - self.pose.x)

    def angular_vel(self):
        omg = self.Kp_ang * normalize_angle(self.steering_angle() - self.pose.theta)
        return saturation(omg, self.ang_vel_min, self.ang_vel_max)

    def stop_turtle(self):
	    # TODO: publish a message to self.vel_pub to stop the turtle completely.
	    # The fields of the message must be set appropriately to stop the turtle.
        self.vel_pub.publish(Twist())

    def move2goal(self):
        """Main function to continuously move the turtle to the goal."""
        vel_msg = Twist()       # The turtle control message

        rospy.loginfo("Started turtle controller.")

        while not rospy.is_shutdown():
            # If not yet started, just wait
            if not self.started:
                self.rate.sleep()
                continue
            
            else: # TODO: Check if goal has been reached
                # Send notification that goal has been reached
                if self.euclidean_distance() < self.distance_tolerance:
                    self.reached_goal = True
                    # TODO: stop the turtle
                    self.stop_turtle()
                    # TODO: publish to '/turtle1ctrl/reach' for reach notification
                    self.reach_pub.publish(Empty())
                    rospy.loginfo("Goal reached: x = %g, y = %g", self.goal_pose.x, self.goal_pose.y)

                else:
                    # Else, control the turtle
                    self.reached_goal = False

                    # TODO: Linear velocity in the x-axis.
                    vel_msg.linear.x = self.linear_vel()
                    vel_msg.linear.y = 0
                    vel_msg.linear.z = 0
                    # TODO: Angular velocity in the z-axis.
                    vel_msg.angular.x = 0
                    vel_msg.angular.y = 0
                    vel_msg.angular.z = self.angular_vel()

                    # TODO: Publish vel_msg to cmd_vel topic
                    self.vel_pub.publish(vel_msg)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping our robot after the movement is over.
        self.stop_turtle()


if __name__ == '__main__':
    try:
        x = MoveToGoal()
        x.move2goal()
    except rospy.ROSInterruptException:
        pass
