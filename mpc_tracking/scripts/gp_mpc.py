#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Twist 
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np
import csv
import sys
sys.path.append(r"casadiinstalldir")
from casadi import *
from numpy import linalg as LA
import tf
import pickle

from GP import sparse_GP

class MPC_tracking:
    """
    The class that handles mpc controller for tracking the waypoints using casadi and ipopt.
    """
    def __init__(self):
        self.T = 0.2; self.H = 2

        self.x = 0; self.y = 0; self.th = 0
        self.v = 0; self.delta = 0

        self.delta_min = -0.4; self.delta_max = 0.4
        self.v_min = 0; self.v_max = 2
        
        self.ramp_v_min = -5*self.T; self.ramp_v_max = 5*self.T
        self.ramp_delta_min = -np.pi*self.T; self.ramp_delta_max = np.pi*self.T


        self.J = 0; # Objective function
        self.g = [] # Constraints
        self.Q = 0; self.R = 0

        self.servo_offset = 0.0
        self.start = 0
        self.idx = 0 

        ## Driving msg
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.header.frame_id = "laser"
        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.speed = 0.0

        drive_topic = '/drive'
        pf_pose_topic = 'pf/viz/inferred_pose'

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped) # Publish to drive
        self.pose_sub = rospy.Subscriber(pf_pose_topic, PoseStamped, self.pose_callback)

        # Loading reference trajectory from csv file
        # input_file = rospy.get_param("~wp_file")
        input_file = "/home/vietanhle/data_file/wp-mpc-200ms.csv"
        x_list = []; y_list = []
        with open(input_file) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                x_list.append(float(line[0])) 
                y_list.append(float(line[1]))
        self.wp_len = len(x_list)
        self.ref = np.empty((2, self.wp_len + self.H))
        for i in range(self.wp_len + self.H):
            if i < self.wp_len:
                self.ref[0,i] = x_list[i]
                self.ref[1,i] = y_list[i] 
            else:
                self.ref[0,i] = x_list[-1]
                self.ref[1,i] = y_list[-1]


        # Load gp model from file, currently using sparse GP model
        m_dict = pickle.load(open("/home/vietanhle/lva_ws/src/mpc_tracking/scripts/sparse_dx.pkl", "rb"))
        self.gp_dx = sparse_GP(m_dict)
        m_dict = pickle.load(open("/home/vietanhle/lva_ws/src/mpc_tracking/scripts/sparse_dy.pkl", "rb"))
        self.gp_dy = sparse_GP(m_dict)
        m_dict = pickle.load(open("/home/vietanhle/lva_ws/src/mpc_tracking/scripts/sparse_dth.pkl", "rb"))
        self.gp_dth = sparse_GP(m_dict)

    def pose_callback(self, msg):
        self.start = 1 # Start MPC controller whenever localization is ready
        self.x = msg.pose.position.x; self.y = msg.pose.position.y

        quaternion = np.array([msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w])

        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.th = euler[2] 

    def set_weight(self, Q, R):
        self.Q = Q; self.R = R
    
    def formulate_mpc(self):
        gp_in = SX.sym('gp_in',4,1)
        mean = self.gp_dx.predict(gp_in)[0]
        f_mean_dx = Function('f_mean_dx', [gp_in], [mean])
        mean = self.gp_dy.predict(gp_in)[0]
        f_mean_dy = Function('f_mean_dy', [gp_in], [mean])
        gp_in = SX.sym('gp_in',2,1)
        mean = self.gp_dth.predict(gp_in)[0]
        f_mean_dth = Function('f_mean_dth', [gp_in], [mean])
        
        x = SX.sym('x'); y = SX.sym('y'); th = SX.sym('th') 
        v = SX.sym('v'); delta = SX.sym('delta') 
        state = np.array([x, y, th])
        control = np.array([v, delta])
        rhs = np.array([f_mean_dx(np.array([np.cos(th), np.sin(th), v, delta])),
            f_mean_dy(np.array([np.cos(th), np.sin(th), v, delta])),
            f_mean_dth(np.array([v, delta]))]) + state

        f_dyn = Function('f_dyn', [state,control], [rhs])
        
        self.opti = casadi.Opti()
        self.U = self.opti.variable(2, self.H)
        self.X = self.opti.variable(3, self.H+1)
        self.P_1 = self.opti.parameter(3)
        self.P_2 = self.opti.parameter(2)
        self.P_3 = self.opti.parameter(2, self.H)

        for k in range(self.H):
            p_H = self.X[0:2, k+1]
            u_H = self.U[:,k]
            self.J += mtimes([(p_H-self.P_3[:,k]).T, self.Q, (p_H-self.P_3[:,k])]) + mtimes([u_H.T, self.R, u_H]) 
            
        self.opti.minimize(self.J) 
        
        for k in range(self.H):
            self.opti.subject_to(self.X[:,k+1] == f_dyn(self.X[:,k], self.U[:,k]))
            
        self.opti.subject_to(self.v_min <= self.U[0,:])
        self.opti.subject_to(self.U[0,:] <= self.v_max)            
        self.opti.subject_to(self.delta_min <= self.U[1,:])
        self.opti.subject_to(self.U[1,:] <= self.delta_max)
        
        # self.opti.subject_to(self.ramp_v_min <= self.U[0,1:] - self.U[0,0:-1])
        # self.opti.subject_to(self.U[0,1:] - self.U[0,0:-1] <= self.ramp_v_max)
        # self.opti.subject_to(self.ramp_delta_min <= self.U[1,1:] - self.U[1,0:-1])
        # self.opti.subject_to(self.U[1,1:] - self.U[1,0:-1] <= self.ramp_delta_max)

        self.opti.subject_to(self.ramp_v_min <= self.U[0,0] - self.P_2[0])
        self.opti.subject_to(self.U[0,0] - self.P_2[0] <= self.ramp_v_max)
        self.opti.subject_to(self.ramp_delta_min <= self.U[1,0] - self.P_2[1])
        self.opti.subject_to(self.U[1,0] - self.P_2[1] <= self.ramp_delta_max)

        self.opti.subject_to(self.X[:,0] == self.P_1[0:3])
        
        p_opts = {'verbose_init': False}
        s_opts = {'tol': 0.01, 'print_level': 0, 'max_iter': 50}
        self.opti.solver('ipopt', p_opts, s_opts)

    def solveMPC(self, ref):
        state = np.array([self.x, self.y, self.th])
        input = np.array([self.v, self.delta])
        self.opti.set_value(self.P_1, state)
        self.opti.set_value(self.P_2, input)
        self.opti.set_value(self.P_3, ref)
        try:
            sol = self.opti.solve()
        except RuntimeError:
            print("An exception occurred")
            control = self.opti.debug.value(self.U)
        else:
            control = sol.value(self.U)
            self.opti.set_initial(self.X, np.hstack((sol.value(self.X)[:,1:], sol.value(self.X)[:,-1:])))
            self.opti.set_initial(self.U, np.hstack((sol.value(self.U)[:,1:], sol.value(self.U)[:,-1:])))
            # self.opti.set_initial(self.X, sol.value(self.X))
            # self.opti.set_initial(self.U, sol.value(self.U))
        self.v = control[0,0]
        self.delta = control[1,0]

    def run(self):
        if self.start == 1 and self.idx < self.wp_len:
            self.solveMPC(self.ref[:, self.idx:self.idx+self.H])
            self.idx += 1
            self.drive_msg.header.stamp = rospy.Time.now()
            self.drive_msg.drive.speed = self.v
            self.drive_msg.drive.steering_angle = self.delta 
            # rospy.loginfo("steer %.3f - speed %1.3f", self.drive_msg.drive.steering_angle, self.drive_msg.drive.speed)
        else:
            self.drive_msg.header.stamp = rospy.Time.now()
            self.drive_msg.drive.steering_angle = 0.0
            self.drive_msg.drive.speed = 0.0 

        self.drive_pub.publish(self.drive_msg)

def main():
    rospy.init_node('gp_mpc_node')
    car = MPC_tracking()
    car.set_weight(1*np.eye(2), np.diag([0 ,2]))
    car.formulate_mpc()
    rate = rospy.Rate(1/car.T)
    while not rospy.is_shutdown():
        car.run()
        rate.sleep()

if __name__ == '__main__':
    main()