#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Twist 
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import csv
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import tf
import sys
sys.path.append(r"casadiinstalldir")
import casadi as ca
import pickle

import atexit
from os.path import expanduser
from time import gmtime, strftime

from GP import sparse_GP

# home = expanduser('~')
# file = open(strftime(home+'/training-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')

def shutdown():
    file.close()

class MPCC:
    """
    The class that handles mpc controller for tracking the waypoints using casadi and ipopt.
    """
    def __init__(self):
        self.x = 0; self.y = 0; self.th = 0
        self.gt_x = 0; self.gt_y = 0; self.gt_th = 0
        self.v = 0; self.delta = 0
        self.arc = 0
        self.state = np.array([self.x, self.y, self.th])
        self.input = np.array([self.v, self.delta])
        
        self.delta_min = -0.4; self.delta_max = 0.4
        self.v_min = 0; self.v_max = 2
        self.nu_min = 0; self.nu_max = 20

        self.T = 0.2; self.H = 10
        self.lf = 0.16; self.lr = 0.17

        self.ramp_v_min = -5*self.T; self.ramp_v_max = 5*self.T
        self.ramp_delta_min = -np.pi*self.T; self.ramp_delta_max = np.pi*self.T

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
        pf_pose_topic = '/pf/viz/inferred_pose'
        gt_pose_topic = '/gt_pose'

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped) # Publish to drive
        self.pose_sub = rospy.Subscriber(pf_pose_topic, PoseStamped, self.pose_callback)
        self.gt_pose_sub = rospy.Subscriber(gt_pose_topic, PoseStamped, self.gt_pose_callback)

        # Load gp model from file, currently using sparse GP model
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dx.pkl", "rb"))
        self.gp_dx = sparse_GP(m_dict)
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dy.pkl", "rb"))
        self.gp_dy = sparse_GP(m_dict)
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dth.pkl", "rb"))
        self.gp_dth = sparse_GP(m_dict)


    def gt_pose_callback(self, msg):
        self.gt_x = msg.pose.position.x
        self.gt_y = msg.pose.position.y

        quaternion = np.array([msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w])

        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.gt_th = euler[2] 

    def pose_callback(self, msg):
        self.start = 1 # Start MPC controller whenever localization is ready
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y

        quaternion = np.array([msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w])

        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.th = euler[2] 

    def set_weight(self, qc, ql, Ru, Rv, gamma):
        self.qc = qc
        self.ql = ql
        self.Ru = Ru
        self.Rv = Rv
        self.gamma = gamma

    def cubic_spline(self, step = 40):
        # Loading waypoints from csv file
        # input_file = rospy.get_param("~wp_file")
        input_file = "/home/lva/data_file/wp-for-mpcc.csv"
        data = genfromtxt(input_file, delimiter=',')
        x_list = data[:, 0] 
        y_list = data[:, 1]
        self.wp_len = len(x_list)
        l_list = np.arange(0, self.wp_len, 1)
        self.L = int(self.wp_len/step)*step
        self.cs_x = ca.interpolant('cs_x','bspline',[l_list[::step]],x_list[::step])
        self.cs_y = ca.interpolant('cs_y','bspline',[l_list[::step]],y_list[::step])
        th = ca.MX.sym('th')
        # Tangent angle
        self.Phi = ca.Function('Phi', [th], [ca.arctan(ca.jacobian(self.cs_y(th),th)/ca.jacobian(self.cs_x(th),th))])
        print(self.L)
    
    def formulateMPC(self):
        X = ca.MX.sym('X'); Y = ca.MX.sym('Y'); th = ca.MX.sym('th')
        self.e_c = ca.Function('e_c', [X, Y, th], [ca.sin(self.Phi(th))*(X - self.cs_x(th)) - ca.cos(self.Phi(th))*(Y - self.cs_y(th))])
        self.e_l = ca.Function('e_l', [X, Y, th], [-ca.cos(self.Phi(th))*(X - self.cs_x(th)) - ca.sin(self.Phi(th))*(Y - self.cs_y(th))]) 
        
        gp_in = ca.SX.sym('gp_in',4,1)
        mean = self.gp_dx.predict(gp_in)[0]
        f_mean_dx = ca.Function('f_mean_dx', [gp_in], [mean])
        mean = self.gp_dy.predict(gp_in)[0]
        f_mean_dy = ca.Function('f_mean_dy', [gp_in], [mean])
        gp_in2 = ca.SX.sym('gp_in2',2,1)
        mean = self.gp_dth.predict(gp_in2)[0]
        f_mean_dth = ca.Function('f_mean_dth', [gp_in2], [mean])
        
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); th = ca.SX.sym('th') 
        v = ca.SX.sym('v'); delta = ca.SX.sym('delta') 
        state = np.array([x, y, th])
        control = np.array([v, delta])
        rhs = np.array([f_mean_dx(np.array([np.cos(th), np.sin(th), v, delta])),
            f_mean_dy(np.array([np.cos(th), np.sin(th), v, delta])),
            f_mean_dth(np.array([v, delta]))]) + state

        self.f_dyn = ca.Function('f_dyn', [state,control], [rhs])
        
        self.mpc_opti = ca.casadi.Opti()
        self.U = self.mpc_opti.variable(2, self.H)
        self.X = self.mpc_opti.variable(3, self.H+1)
        self.TH = self.mpc_opti.variable(self.H+1)
        self.NU = self.mpc_opti.variable(self.H)
        self.P_1 = self.mpc_opti.parameter(3)
        self.P_2 = self.mpc_opti.parameter(2)
        
        J = 0
        for k in range(self.H+1):
            J += self.qc*self.e_c(self.X[0,k], self.X[1,k], self.TH[k])**2 + self.ql*self.e_l(self.X[0,k], self.X[1,k], self.TH[k])**2
        for k in range(self.H-1):
            J += self.Ru[0]*(self.U[0,k+1]-self.U[0,k])**2 + self.Ru[1]*(self.U[1,k+1]-self.U[1,k])**2 + self.Rv*(self.NU[k+1]-self.NU[k])**2
            
        J += -self.gamma*self.TH[-1]
        self.mpc_opti.minimize(J) 
        
        for k in range(self.H):
            self.mpc_opti.subject_to(self.X[:,k+1] == self.f_dyn(self.X[:,k], self.U[:,k]))
            self.mpc_opti.subject_to(self.TH[k+1] == self.TH[k] + self.T*self.NU[k])
            
        self.mpc_opti.subject_to(0 <= self.TH)
        self.mpc_opti.subject_to(self.TH <= self.L)
        
        self.mpc_opti.subject_to(self.nu_min <= self.NU)
        self.mpc_opti.subject_to(self.NU <= self.nu_max)
        
        self.mpc_opti.subject_to(self.v_min <= self.U[0,:])
        self.mpc_opti.subject_to(self.U[0,:] <= self.v_max)
        self.mpc_opti.subject_to(self.delta_min <= self.U[1,:])
        self.mpc_opti.subject_to(self.U[1,:] <= self.delta_max)
        
        self.mpc_opti.subject_to(self.ramp_v_min <= self.U[0,0] - self.P_2[0])
        self.mpc_opti.subject_to(self.U[0,0] - self.P_2[0] <= self.ramp_v_max)
        self.mpc_opti.subject_to(self.ramp_delta_min <= self.U[1,0] - self.P_2[1])
        self.mpc_opti.subject_to(self.U[1,0] - self.P_2[1] <= self.ramp_delta_max)
        
        self.mpc_opti.subject_to(self.ramp_v_min <= self.U[0,1:] - self.U[0,0:-1])
        self.mpc_opti.subject_to(self.U[0,1:] - self.U[0,0:-1] <= self.ramp_v_max)
        self.mpc_opti.subject_to(self.ramp_delta_min <= self.U[1,1:] - self.U[1,0:-1])
        self.mpc_opti.subject_to(self.U[1,1:] - self.U[1,0:-1] <= self.ramp_delta_max)

        self.mpc_opti.subject_to(self.X[:,0] == self.P_1[0:3])
        
        p_opts = {'verbose_init': False}
        s_opts = {'tol': 0.01, 'print_level': 0, 'max_iter': 100}
        self.mpc_opti.solver('ipopt', p_opts, s_opts)

        # Warm up
        self.mpc_opti.set_value(self.P_1, self.state)
        self.mpc_opti.set_value(self.P_2, self.input)
        
        sol = self.mpc_opti.solve()
        self.mpc_opti.set_initial(self.U, sol.value(self.U))
        self.mpc_opti.set_initial(self.X, sol.value(self.X))
        self.mpc_opti.set_initial(self.NU, sol.value(self.NU))
        self.mpc_opti.set_initial(self.TH, sol.value(self.TH))

    def solveMPC(self):
        self.state = np.array([self.x, self.y, self.th])
        self.input = np.array([self.v, self.delta])
        self.mpc_opti.set_value(self.P_1, self.state)
        self.mpc_opti.set_value(self.P_2, self.input)
        try:
            sol = self.mpc_opti.solve()
            
        except RuntimeError:
            print("An exception occurred")
            control = self.mpc_opti.debug.value(self.U)
            self.arc = self.mpc_opti.debug.value(self.TH)[0]
        else:
            control = sol.value(self.U)
            self.mpc_opti.set_initial(self.X, np.hstack((sol.value(self.X)[:,1:], sol.value(self.X)[:,-1:])))
            self.mpc_opti.set_initial(self.U, np.hstack((sol.value(self.U)[:,1:], sol.value(self.U)[:,-1:])))    
            self.mpc_opti.set_initial(self.TH, np.hstack((sol.value(self.TH)[1:], sol.value(self.TH)[-1:])))
            self.mpc_opti.set_initial(self.NU, np.hstack((sol.value(self.NU)[1:], sol.value(self.NU)[-1:]))) 
            self.arc = sol.value(self.TH)[0]
        self.v = control[0,0]
        self.delta = control[1,0]
        print(self.arc)

    def run(self):
        if self.start == 1 and self.arc < self.L - 80:
            self.solveMPC()
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
    # atexit.register(shutdown)
    rospy.init_node('mpc_node')
    car = MPCC()
    car.set_weight(1, 100, np.array([0.1, 10]), 2, 2)
    car.cubic_spline(step = 40)
    car.formulateMPC()
    rate = rospy.Rate(1/car.T)
    while not rospy.is_shutdown():
        car.run()
        # file.write('%f, %f, %f, %f, %f, %f, %f, %f\n' % (car.x, car.y, car.th, car.v, car.delta, car.gt_x, car.gt_y, car.gt_th))
        rate.sleep()

if __name__ == '__main__':
    main()
