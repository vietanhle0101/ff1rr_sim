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
from numpy import genfromtxt

from GP import linsparse_GP

def vertical_reshape(v):
    n_row, n_col = np.shape(v)
    output = v[:,0].reshape(-1) 
    for i in range(1, n_col):
        output = np.hstack([output, v[:,i].reshape(-1)])
    return output

class MPC_tracking:
    """
    The class that handles mpc controller for tracking the waypoints using casadi and ipopt.
    """
    def __init__(self):
        self.T = 0.2; self.H = 5

        self.x = 0; self.y = 0; self.th = 0
        self.v = 0; self.delta = 0

        self.de_min = -0.4; self.de_max = 0.4
        self.v_min = 0; self.v_max = 2
        
        self.ramp_v_min = -5*self.T; self.ramp_v_max = 5*self.T
        self.ramp_de_min = -np.pi*self.T; self.ramp_de_max = np.pi*self.T

        self.J = 0; # Objective function
        self.g = [] # Constraints
        self.Q = 0; self.R = 0

        # Store nominal values, m_hat vectors and control over a horizon 
        self.v_nom = np.zeros(self.H); self.de_nom = np.zeros(self.H)
        self.x_nom = np.zeros(self.H+1); self.y_nom = np.zeros(self.H+1); self.th_nom = np.zeros(self.H+1) 
        self.m_dx = np.zeros((self.H,4)); self.m_dy = np.zeros((self.H,4)); self.m_dth = np.zeros((self.H,3))
        self.v_H = np.zeros(self.H); self.de_H = np.zeros(self.H)

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
        
        # Load reference trajectory from csv file
        input_file = "/home/lva/data_file/data-for-mpc.csv"
        data = genfromtxt(input_file, delimiter=',')
        step = 8
        x_list = data[::step, 0] 
        y_list = data[::step, 1]
        self.wp_len = len(x_list)
        self.ref = np.hstack([np.vstack([x_list, y_list]), [[x_list[-1]], [y_list[-1]]]*np.ones(self.H)])

        # Load gp model (dictionary) from file, using linsparse GP model
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dx.pkl", "rb"))
        self.gp_dx = linsparse_GP(m_dict)
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dy.pkl", "rb"))
        self.gp_dy = linsparse_GP(m_dict)
        m_dict = pickle.load(open("/home/lva/catkin_ws/src/mpc_tracking/scripts/sparse_dth.pkl", "rb"))
        self.gp_dth = linsparse_GP(m_dict)

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

    def objective(self, x, y, v, de, x_r, y_r):
        # Calculate objective function, for both numerical and symbolic variables
        obj = 0
        for k in range (self.H):
            obj += self.Q[0]*(x[k+1] - x_r[k])**2 + self.Q[1]*(y[k+1] - y_r[k])**2 + self.R[0]*v[k]**2 + self.R[1]*de[k]**2
        return obj  
        
    def update_linGP(self, v, de):
        # Update linGP model (m_hat for mean) for dth, dx, dy, and set nominal values
        x = np.zeros(self.H+1); y = np.zeros(self.H+1); th = np.zeros(self.H+1) 
        x[0] = self.x; y[0] = self.y; th[0] = self.th

        for k in range(self.H):
            xk = np.array([v[k], de[k]]) 
            m_dth = self.gp_dth.linearize(xk)
            self.m_dth[k] = m_dth.reshape((-1,))
            th[k+1] = th[k] + m_dth[0]
        
        for k in range(self.H):
            xk = np.array([np.cos(th[k]), np.sin(th[k]), v[k], de[k]]) 
            m_dx = self.gp_dx.linearize(xk)            
            m_dy = self.gp_dy.linearize(xk)
            x[k+1] = x[k] + m_dx[0]
            y[k+1] = y[k] + m_dy[0]
            M = np.array([np.hstack([1, np.zeros(4)]), np.hstack([0, -xk[1], xk[0], np.zeros(2)]), 
              np.hstack([np.zeros(3), 1, 0]), np.hstack([np.zeros(4), 1])])
            self.m_dx[k] = np.dot(M, m_dx.reshape((-1,)))
            self.m_dy[k] = np.dot(M, m_dy.reshape((-1,))) 
            
        self.v_nom = v; self.de_nom = de
        self.x_nom = x; self.y_nom = y; self.th_nom = th    

    def simulate_linGP(self, dv, dde):
        # Simulate linGP model over a horizon
        dx = np.zeros(self.H); dy = np.zeros(self.H); dth = np.zeros(self.H)
        x = np.zeros(self.H+1); y = np.zeros(self.H+1); th = np.zeros(self.H+1) 
        x[0] = self.x; y[0] = self.y; th[0] = self.th

        for k in range(self.H):
            dth[k] = self.m_dth[k,0] + self.m_dth[k,1]*dv[k] + self.m_dth[k,2]*dde[k]
            th[k+1] = th[k] + dth[k]
        
        for k in range(self.H):
            dx[k] = self.m_dx[k,0] + self.m_dx[k,1]*(th[k] - self.th_nom[k]) + self.m_dx[k,2]*dv[k] + self.m_dx[k,3]*dde[k]
            dy[k] = self.m_dy[k,0] + self.m_dy[k,1]*(th[k] - self.th_nom[k]) + self.m_dy[k,2]*dv[k] + self.m_dy[k,3]*dde[k]
            x[k+1] = x[k] + dx[k]
            y[k+1] = y[k] + dy[k]
        
        return x, y, th

    def simulate_GP(self, v, de):
        # Simulate GP model over a horizon
        x = np.zeros(self.H+1); y = np.zeros(self.H+1); th = np.zeros(self.H+1) 
        x[0] = self.x; y[0] = self.y; th[0] = self.th
        for k in range(self.H):
            xk = np.array([v[k], de[k]]) 
            th[k+1] = th[k] + self.gp_dth.predict(xk)[0]
        
        for k in range(self.H):
            xk = np.array([np.cos(th[k]), np.sin(th[k]), v[k], de[k]]) 
            x[k+1] = x[k] + self.gp_dx.predict(xk)[0]  
            y[k+1] = y[k] + self.gp_dy.predict(xk)[0]
            
        return x, y, th

    def SCP_linGP(self, i, maxiters = 10, rho_min = 0.0, rho_max = inf, r0 = 0.0, r1 = 0.1, r2 = 0.2, b_fail = 0.5, b_succ = 2.0, thres = 0.001):
        self.rho = 0.5
        self.update_linGP(np.hstack([self.v_H[1:], self.v_H[0]]), np.hstack([self.de_H[1:], self.de_H[0]]))
        x_ref = self.ref[0,i:i+self.H]; y_ref = self.ref[1,i:i+self.H]
        J_exact = self.objective(self.x_nom, self.y_nom, self.v_nom, self.de_nom, x_ref, y_ref)
        for j in range(maxiters):
            dv_sol, dde_sol, v_sol, de_sol = self.solve_lingpmpc(i)
            x_bar, y_bar, th_bar = self.simulate_GP(v_sol, de_sol)
            x_til, y_til, th_til = self.simulate_linGP(dv_sol, dde_sol)
            J_bar = self.objective(x_bar, y_bar, v_sol, de_sol, x_ref, y_ref)
            J_til = self.objective(x_til, y_til, v_sol, de_sol, x_ref, y_ref)
            dJ_bar = J_exact - J_bar; dJ_til = J_exact - J_til
            if abs(dJ_til) < thres: # stop and return solution
                return v_sol, de_sol
            else:
                ratio = dJ_bar/dJ_til
                if ratio > r0: # Accept solution
                    self.update_linGP(v_sol, de_sol)
                    J_exact = J_bar
                    if ratio < r1: self.rho *= b_fail
                    elif ratio > r2: self.rho *= b_succ
                else: # Keep current solution 
                    self.rho *= b_fail
                self.rho = max(rho_min, min(rho_max, self.rho))
        return v_sol, de_sol

    def formulate_mpc(self):
        self.dv = SX.sym('dv', self.H); self.dde = SX.sym('dde', self.H) # will be optimization variables, currently using single shooting
        self.dx = SX.zeros(self.H); self.dy = SX.zeros(self.H); self.dth = SX.zeros(self.H)
        self.x_h = SX.zeros(self.H+1); self.y_h = SX.zeros(self.H+1); self.th_h = SX.zeros(self.H+1)  
        # Symbolic parameters
        self.p1 = SX.sym('p1', 3)
        self.p2 = SX.sym('p2', 2)
        self.px_r = SX.sym('px_r', self.H); self.py_r = SX.sym('py_r', self.H)
        self.pv_n = SX.sym('pv_n', self.H); self.pde_n = SX.sym('pde_n', self.H) 
        self.pth_n = SX.sym('pth_n', self.H+1)
        self.pm_dth = SX.sym('pm_dth', self.H, 3)
        self.pm_dx = SX.sym('pm_dx', self.H, 4)
        self.pm_dy = SX.sym('pm_dy', self.H, 4)

        self.v_h = self.pv_n + self.dv; self.de_h = self.pde_n + self.dde        
        self.x_h[0] = self.p1[0]; self.y_h[0] = self.p1[1]; self.th_h[0] = self.p1[2]

        for k in range(self.H):
            self.dth[k] = self.pm_dth[k,0] + self.pm_dth[k,1]*self.dv[k] + self.pm_dth[k,2]*self.dde[k]
            self.th_h[k+1] = self.th_h[k] + self.dth[k]
            
        for k in range(self.H):
            self.dx[k] = self.pm_dx[k,0] + self.pm_dx[k,1]*(self.th_h[k] - self.pth_n[k]) + self.pm_dx[k,2]*self.dv[k] + self.pm_dx[k,3]*self.dde[k]
            self.dy[k] = self.pm_dy[k,0] + self.pm_dy[k,1]*(self.th_h[k] - self.pth_n[k]) + self.pm_dy[k,2]*self.dv[k] + self.pm_dy[k,3]*self.dde[k]
            self.x_h[k+1] = self.x_h[k] + self.dx[k] 
            self.y_h[k+1] = self.y_h[k] + self.dy[k]
            
        self.J = self.objective(self.x_h, self.y_h, self.v_h, self.de_h, self.px_r, self.py_r) 
        self.g = []
        self.ubg = []; self.lbg = []
        
        # Construct constraints
        for k in range(self.H):
            self.lbg += [self.v_min, self.de_min]; self.ubg += [self.v_max, self.de_max]  
            self.g += [self.v_h[k]]; self.g += [self.de_h[k]]
            self.lbg += [self.ramp_v_min, self.ramp_de_min]; self.ubg += [self.ramp_v_max, self.ramp_de_max]  
            if k == 0:
                self.g += [self.v_h[k] - self.p2[0]]; self.g += [self.de_h[k] - self.p2[1]]
            else:
                self.g += [self.v_h[k] - self.v_h[k-1]]; self.g += [self.de_h[k] - self.de_h[k-1]]   
        
        self.qp = {'x':vertcat(self.dv, self.dde), 'f':self.J, 'g': vertcat(*self.g), 'p': vertcat(self.p1, self.p2, self.px_r, self.py_r, self.pv_n, self.pde_n, self.pth_n, \
            reshape(self.pm_dth, -1, 1), reshape(self.pm_dx, -1, 1), reshape(self.pm_dy, -1, 1))}
        self.S = qpsol('S', 'qpoases', self.qp, {'printLevel':'none'})

        # Run a trial to warm up
        self.rho = 0.1 # To run a trial
        self.lbx = [-self.rho]*self.H + [-self.rho]*self.H; self.ubx = [self.rho]*self.H + [self.rho]*self.H
        self.p = [self.x, self.y, self.th] + [self.v, self.delta] + self.ref[0,0:0+self.H].tolist() + self.ref[1,0:0+self.H].tolist() + self.v_nom.tolist() + self.de_nom.tolist() + self.th_nom.tolist() \
                + vertical_reshape(self.m_dth).tolist() + vertical_reshape(self.m_dx).tolist()+ vertical_reshape(self.m_dy).tolist()
        sol = self.S(lbg = self.lbg, ubg = self.ubg, lbx = self.lbx, ubx = self.ubx, p = self.p)


    def solve_lingpmpc(self, i):  
        self.lbx = [-self.rho]*self.H + [-self.rho]*self.H; self.ubx = [self.rho]*self.H + [self.rho]*self.H # Trust region constraint
        self.p = [self.x, self.y, self.th] + [self.v, self.delta] + self.ref[0,i:i+self.H].tolist() + self.ref[1,i:i+self.H].tolist() + self.v_nom.tolist() + self.de_nom.tolist() + self.th_nom.tolist() \
                + vertical_reshape(self.m_dth).tolist() + vertical_reshape(self.m_dx).tolist()+ vertical_reshape(self.m_dy).tolist()
        sol = self.S(lbg = self.lbg, ubg = self.ubg, lbx = self.lbx, ubx = self.ubx, p = self.p)
        sol = sol['x']
        dv_sol = sol[0:self.H].full().reshape(self.H) 
        dde_sol = sol[self.H:].full().reshape(self.H)
        v_sol = dv_sol + self.v_nom
        de_sol = dde_sol + self.de_nom

        return dv_sol, dde_sol, v_sol, de_sol

    def control(self, i):
        start = rospy.get_time()
        self.v_H, self.de_H = self.SCP_linGP(i)
        end = rospy.get_time()
        print(i, end - start)
        self.v = self.v_H[0]; self.delta = self.de_H[0]

    def run(self):
        if self.start == 1 and self.idx < self.wp_len:
            self.control(self.idx)
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
    car.set_weight(np.array([5,5]), np.array([0.1, 10]))
    car.formulate_mpc()
    rate = rospy.Rate(1/car.T)
    while not rospy.is_shutdown():
        car.run()
        rate.sleep()

if __name__ == '__main__':
    main()