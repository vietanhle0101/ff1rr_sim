#!/usr/bin/env python
import rospy
import numpy as np
import atexit
import tf
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

home = expanduser('~')
file = open(strftime(home+'/wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')

def save_waypoint(data):
    quaternion = np.array([data.pose.orientation.x, 
                           data.pose.orientation.y, 
                           data.pose.orientation.z, 
                           data.pose.orientation.w])

    euler = tf.transformations.euler_from_quaternion(quaternion)
    speed = LA.norm(np.array([data.pose.position.x, 
                              data.pose.position.y, 
                              data.pose.position.z]),2)
    # if data.twist.twist.linear.x>0.:
    #     print data.twist.twist.linear.x

    file.write('%f, %f, %f, %f\n' % (data.pose.position.x,
                                     data.pose.position.y,
                                     euler[2],
                                     speed))

def shutdown():
    file.close()
    print('Goodbye')
 
def listener():
    rospy.init_node('waypoints_logger', anonymous=True)
    rospy.Subscriber('pf/viz/inferred_pose', PoseStamped, save_waypoint)
    rospy.spin()
    # rate = rospy.Rate(1)
    # rate.sleep()


if __name__ == '__main__':
    atexit.register(shutdown)
    print('Saving waypoints...')

    try:
        listener()
    except rospy.ROSInterruptException:
        pass
