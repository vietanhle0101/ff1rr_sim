#!/usr/bin/env python
import rospy

# Import message types
from turtlesim.msg import Pose
from std_msgs.msg import Empty  # Empty message for notification of reaching goal

# Import turtlesim services
import turtlesim.srv
import std_srvs.srv

# Some math functions for the computation
from math import sin, cos, pi, atan2

class FigureDrawer:
	"""Use the turtle to draw a figure given by a Figure object."""	
	def __init__(self, fig):
		# TODO: Create a node named 'figure_drawer' that will provide the trajectory for the turtle controller.
		rospy.init_node('figure_drawer')

		self.figure = fig
		# TODO: Publisher for the goal, to 'turtle1ctrl/goal'
		self.goal_pub = rospy.Publisher('turtle1ctrl/goal', Pose)
		# TODO: Subscriber to 'turtle1ctrl/reach' for goal reaching notification
		# self.reached_goal is the callback function
		self.reach_sub = rospy.Subscriber('turtle1ctrl/reach', Empty, self.reached_goal)
		self.goal = Pose()      # The goal message
		# Pen color [R, G, B]
		self.pen = [200, 0, 0]
		self.penwidth = 2

	def reset(self):
		# Reset turtlesim simulator
		rospy.wait_for_service('reset')
		try:
			reset_turtle = rospy.ServiceProxy('reset', std_srvs.srv.Empty)
			reset_turtle()
		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s", e)

	def set_pen(self, penoff):
		# Call the service "set_pen" of turtle
		rospy.wait_for_service('turtle1/set_pen')
		try:
			set_pen_srv = rospy.ServiceProxy('turtle1/set_pen', turtlesim.srv.SetPen)
			set_pen_srv(self.pen[0], self.pen[1], self.pen[2], self.penwidth, penoff)
		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s", e)

	def initialize(self):
		# Move the turtle to the initial position and get the pen ready
		# Turn off pen
		self.set_pen(1)

		# Get initial position and move there
		rospy.wait_for_service('turtle1/teleport_absolute')
		try:
			move_turtle = rospy.ServiceProxy('turtle1/teleport_absolute', turtlesim.srv.TeleportAbsolute)
			x0, y0, theta0 = self.figure.init_pos()
			move_turtle(x0, y0, theta0)
		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s", e)

		# TODO: Turn on pen
		self.set_pen(0)

	def reached_goal(self, data):
		"""Callback to set the next goal."""
		self.goal.x, self.goal.y = self.figure.next()
		# TODO: publish the next goal
		self.goal_pub.publish(self.goal)
			
	def draw(self):
		"""Draw the figure."""
		# Reset the turtle simulator
		self.reset()
		# Initialize the turlte
		self.initialize()
		# Sleep a bit for the position to be published to other node
		rospy.sleep(.5)
		# Publish the first goal
		self.goal.x, self.goal.y = self.figure.next()
		# TODO: publish the goal
		self.goal_pub.publish(self.goal)

		# Let it run
		rospy.spin()

class FigureCosSin:
	"""x = r*cos(a*t), y = r*sin(b*t). Try (a=1,b=2) (figure 8), (a=3,b=5)."""
	def __init__(self, a, b, r, N):
		self.a = a
		self.b = b
		self.r = r
		self.k = 0              # step counter
		self.step = 2*pi/N

		# Offsets for the coordinates
		self.x_offset = 5.
		self.y_offset = 5.

	def calc_pos(self, k):
		"""Calculate position for step k."""
		# TODO: calculate the waypoint at index k,
		# using the equations from the lab instructions
		t = k*self.step
		x = self.r*cos(self.a*t) + self.x_offset
		y = self.r*sin(self.b*t) + self.y_offset
		return (x, y)

	def init_pos(self):
		"""Initial position. The angle theta will point from k=0 to k=1."""
		x0, y0 = self.calc_pos(0)
		x1, y1 = self.calc_pos(1)
		theta0 = atan2(y1-y0, x1-x0)
		return (x0, y0, theta0)

	def next(self):
		"""Return the next point and increase the step counter."""
		self.k += 1
		return self.calc_pos(self.k)

	
if __name__ == '__main__':
    try:
	    # TODO: create a FigureCosSin object with suitable step size parameter N
		fig = FigureCosSin(1.0, 2.0, 4.0, 100)
		drawer = FigureDrawer(fig)
		rospy.loginfo("Start Drawing Figure")
		drawer.draw()
    except rospy.ROSInterruptException:
        pass
