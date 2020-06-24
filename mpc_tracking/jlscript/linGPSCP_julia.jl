#!/usr/bin/env julia

using RobotOS
@rosimport geometry_msgs.msg: PoseStamped, Twist
@rosimport sensor_msgs.msg: LaserScan
@rosimport ackermann_msgs.msg: AckermannDriveStamped

rostypegen()
using .geometry_msgs.msg
using .sensor_msgs.msg
using .ackermann_msgs.msg

using PyCall
@pyimport tf

using LinearAlgebra
using JLD
using CSV
using JuMP
import OSQP
include("linsparseGP.jl")

mutable struct linGP_SCP
    T; H;
    gp_dx; gp_dy; gp_dth # GP model
    x0; y0; th0; v0; de0 # The current states and inputs of the car
    v_min; v_max; de_min; de_max
    ramp_v_min; ramp_v_max; ramp_de_min; ramp_de_max
    Q; R; Rd; # Weight matrices
    xref; yref # Reference in a horizon
    v_nom; de_nom; x_nom; y_nom; th_nom  # The current nominal states and inputs
    m_dx; m_dy; m_dth # m_hat matrices
    v_H; de_H     # Control signals over MPC horizon
    model::JuMP.Model  # The model for MPC problem
    vars; rho
    start

    function linGP_SCP(H::Integer, gp_dx::linsparse_GP, gp_dy::linsparse_GP, gp_dth::linsparse_GP)
        T = 0.2
        obj = new(T, H, gp_dx, gp_dy, gp_dth, 0., 0., 0., 0., 0.)  # Incomplete init
        obj.v_min = 0; obj.v_max = 2
        obj.de_min = -0.4; obj.de_max = 0.4
        obj.ramp_v_min = -5*T; obj.ramp_v_max = 5*T;
        obj.ramp_de_min = -pi*T; obj.ramp_de_max = pi*T
        obj.v_nom = zeros(H); obj.de_nom = zeros(H)
        obj.x_nom = zeros(H); obj.y_nom = zeros(H); obj.th_nom = zeros(H)
        obj.m_dx = zeros(H,4); obj.m_dy = zeros(H,4); obj.m_dth = zeros(H,3)
        obj.v_H = zeros(H); obj.de_H = zeros(H)
        obj.start = 0
        return obj
    end
end

function set_inputs!(c::linGP_SCP, u::AbstractVector)
    c.v0 = u[1]; c.de0 = u[2]
end

function set_refs!(c::linGP_SCP, xref::AbstractVector, yref::AbstractVector)
    c.xref = xref; c.yref = yref;
end

function set_params!(c::linGP_SCP, Q::AbstractVector, R::AbstractVector, Rd::AbstractVector)
    c.Q = Q; c.R = R; c.Rd = Rd;
end

function objective(c::linGP_SCP, x::AbstractVector, y::AbstractVector, v::AbstractVector, de::AbstractVector,
     x_r::AbstractVector, y_r::AbstractVector)
    obj = 0
    for k = 1:c.H
        obj += c.Q[1]*(x[k+1] - x_r[k])^2 + c.Q[2]*(y[k+1] - y_r[k])^2 + c.R[1]*v[k]^2 + c.R[2]*de[k]^2
    end
    for k = 1:c.H-1
        obj += c.Rd[1]*(v[k+1]-v[k])^2 + c.Rd[2]*(de[k+1]-de[k])^2
    end
    return obj
end

function update_linGP(c::linGP_SCP, v::AbstractVector, de::AbstractVector)
    # Update linGP model (m_hat for mean) for dth and dx, dy
    x = zeros(c.H+1); y = zeros(c.H+1); th = zeros(c.H+1)
    x[1] = c.x0; y[1] = c.y0; th[1] = c.th0;
    for k = 1:c.H
        xk = [v[k], de[k]]
        m_dth = linearize(c.gp_dth, xk)
        c.m_dth[k,:] = m_dth
        th[k+1] = th[k] + m_dth[1]
    end
    for k = 1:c.H
        xk = [cos(th[k]), sin(th[k]), v[k], de[k]]
        m_dx = linearize(c.gp_dx, xk)
        m_dy = linearize(c.gp_dy, xk)
        x[k+1] = x[k] + m_dx[1]
        y[k+1] = y[k] + m_dy[1]
        M = [1 zeros(1,4); 0 -xk[2] xk[1] zeros(1, 2); zeros(2, 3) Matrix(I, 2, 2)]
        c.m_dx[k,:] = M*m_dx
        c.m_dy[k,:] = M*m_dy
    end
    c.v_nom = v; c.de_nom = de
    c.x_nom = x; c.y_nom = y; c.th_nom = th
end

function simulate_linGP(c::linGP_SCP, dv::AbstractVector, dde::AbstractVector)
    dx = zeros(c.H); dy = zeros(c.H); dth = zeros(c.H)
    x = zeros(c.H+1); y = zeros(c.H+1); th = zeros(c.H+1)
    x[1] = c.x0; y[1] = c.y0; th[1] = c.th0;
    for k = 1:c.H
        dth[k] = c.m_dth[k,1] + c.m_dth[k,2]*dv[k] + c.m_dth[k,3]*dde[k]
        th[k+1] = th[k] + dth[k]
    end
    for k = 1:c.H
        dx[k] = c.m_dx[k,1] + c.m_dx[k,2]*(th[k] - c.th_nom[k]) + c.m_dx[k,3]*dv[k] + c.m_dx[k,4]*dde[k]
        dy[k] = c.m_dy[k,1] + c.m_dy[k,2]*(th[k] - c.th_nom[k]) + c.m_dy[k,3]*dv[k] + c.m_dy[k,4]*dde[k]
        x[k+1] = x[k] + dx[k]
        y[k+1] = y[k] + dy[k]
    end

    return x, y, th
end

function simulate_GP(c::linGP_SCP, v::AbstractVector, de::AbstractVector)
    x = zeros(c.H+1); y = zeros(c.H+1); th = zeros(c.H+1)
    x[1] = c.x0; y[1] = c.y0; th[1] = c.th0;
    for k = 1:c.H
        xk = [v[k], de[k]]
        th[k+1] =  th[k] + predict(c.gp_dth, xk)[1]
    end
    for k = 1:c.H
        xk = [cos(th[k]), sin(th[k]), v[k], de[k]]
        x[k+1] = x[k] + predict(c.gp_dx, xk)[1]
        y[k+1] = y[k] + predict(c.gp_dy, xk)[1]
    end
    return x, y, th
end

function formulate_linGPMPC(c::linGP_SCP, solver = OSQP.Optimizer)
    c.model = JuMP.Model(JuMP.with_optimizer(solver))
    set_silent(c.model)

    # Variables
    @variable(c.model, dv[1:c.H])
    @variable(c.model, dde[1:c.H])
    @variable(c.model, v[1:c.H])
    @variable(c.model, de[1:c.H])
    @variable(c.model, dth[1:c.H])
    @variable(c.model, dx[1:c.H])
    @variable(c.model, dy[1:c.H])
    @variable(c.model, th[1:c.H+1])
    @variable(c.model, x[1:c.H+1])
    @variable(c.model, y[1:c.H+1])

    J = objective(c, x, y, v, de, c.xref, c.yref);
    @objective(c.model, Min, J);
    @constraints(c.model, begin
        c.v_min .<= v .<= c.v_max
        c.de_min .<= de .<= c.de_max
        c.ramp_v_min .<= v - [c.v0; v[1:end-1]] .<= c.ramp_v_max
        c.ramp_de_min .<= de - [c.de0; de[1:end-1]] .<= c.ramp_de_max
        v - dv - c.v_nom .== 0
        de - dde - c.de_nom .== 0
        th[1] - c.th0 == 0; x[1] - c.x0 == 0; y[1] - c.y0 == 0;
        th[2:end] - cumsum(dth) .== c.th0
        x[2:end] - cumsum(dx) .== c.x0
        y[2:end] - cumsum(dy) .== c.y0
    end)
    for k = 1:c.H
        @constraint(c.model, dth[k] - (c.m_dth[k,1] + c.m_dth[k,2]*dv[k] + c.m_dth[k,3]*dde[k]) == 0)
        @constraint(c.model, dx[k] - (c.m_dx[k,1] + c.m_dx[k,2]*(th[k] - c.th_nom[k]) + c.m_dx[k,3]*dv[k] + c.m_dx[k,4]*dde[k]) == 0)
        @constraint(c.model, dy[k] - (c.m_dy[k,1] + c.m_dy[k,2]*(th[k] - c.th_nom[k]) + c.m_dy[k,3]*dv[k] + c.m_dy[k,4]*dde[k]) == 0)
    end
    set_lower_bound.(dv, -c.rho)
    set_upper_bound.(dv, c.rho)
    set_lower_bound.(dde, -c.rho)
    set_upper_bound.(dde, c.rho)
    c.vars = [dv dde v de]
end

function solve_linGPMPC(c::linGP_SCP)
    JuMP.optimize!(c.model)
    dv_sol = value.(c.vars)[:,1]
    dde_sol = value.(c.vars)[:,2]
    v_sol = value.(c.vars)[:,3]
    de_sol = value.(c.vars)[:,4]

    return dv_sol, dde_sol, v_sol, de_sol
end

function SCP_linGP(c::linGP_SCP, maxiters = 50, rho_min = 0.0, rho_max = 10.0,
                  r0 = 0.0, r1 = 0.1, r2 = 0.2, b_fail = 0.5, b_succ = 2.0, thres = 1e-3)
    c.rho = 0.5
    update_linGP(c, [c.v_H[2:end]; c.v_H[end]], [c.de_H[2:end]; c.de_H[end]])
    J_exact = objective(c, c.x_nom, c.y_nom, c.v_nom, c.de_nom, c.xref, c.yref)
    v_sol = zeros(c.H); de_sol = zeros(c.H)
    for j = 1:maxiters
        formulate_linGPMPC(c)
        dv_sol, dde_sol, v_sol, de_sol = solve_linGPMPC(c)
        x_bar, y_bar, th_bar = simulate_GP(c, v_sol, de_sol)
        x_til, y_til, th_til = simulate_linGP(c, dv_sol, dde_sol)
        J_bar = objective(c, x_bar, y_bar, v_sol, de_sol, c.xref, c.yref)
        J_til = objective(c, x_til, y_til, v_sol, de_sol, c.xref, c.yref)
        dJ_bar = J_exact - J_bar; dJ_til = J_exact - J_til
        if abs(dJ_til/J_exact) < thres # stop and return solution
            return v_sol, de_sol
        else
            ratio = dJ_bar/dJ_til
            if ratio > r0 # Accept solution
                update_linGP(c, v_sol, de_sol)
                J_exact = J_bar
                if ratio < r1
                    c.rho *= b_fail
                elseif ratio > r2
                    c.rho *= b_succ
                end
            else # Keep current solution
                c.rho *= b_fail
            end
            c.rho = max(rho_min, min(rho_max, c.rho))
        end
    end
    return v_sol, de_sol
end

function control(c::linGP_SCP)
    c.v_H, c.de_H = SCP_linGP(c)
    return c.v_H[1], c.de_H[1]
end

function pose_callback(msg::PoseStamped, c::linGP_SCP)
    c.start = 1
    c.x0 = msg.pose.position.x; c.y0 = msg.pose.position.y
    quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
    euler = tf.transformations.euler_from_quaternion(quaternion)
    c.th0 = euler[3]
end

function main()
    # Load neccessary data: reference trajectory and GP data
    push!(LOAD_PATH, ".")
    input_file = "/home/lva/data_file/data-for-mpc.csv"
    df = CSV.read(input_file, header = 0)
    data = convert(Matrix{Float64}, df)
    x_list = data[1:8:end, 1]
    y_list = data[1:8:end, 2]
    L = length(x_list)

    gp_dx_dict = load("/home/lva/catkin_ws/src/mpc_tracking/src/sparse_dx.jld")["data"]
    gp_dy_dict = load("/home/lva/catkin_ws/src/mpc_tracking/src/sparse_dy.jld")["data"]
    gp_dth_dict = load("/home/lva/catkin_ws/src/mpc_tracking/src/sparse_dth.jld")["data"]

    gp_dx = linsparse_GP(gp_dx_dict)
    pre_compute(gp_dx)
    gp_dy = linsparse_GP(gp_dy_dict)
    pre_compute(gp_dy)
    gp_dth = linsparse_GP(gp_dth_dict)
    pre_compute(gp_dth)

    # Initialize a node
    init_node("rosjl_lingpmpc")
    # Initialize linGPSCP struct
    c = linGP_SCP(5, gp_dx, gp_dy, gp_dth)
    # Subscriber and Publisher
    pf_pose_topic = "pf/viz/inferred_pose"
    pose_sub = Subscriber{PoseStamped}(pf_pose_topic, pose_callback, (c,), queue_size = 10)
    drive_topic = "/drive"
    drive_pub = Publisher{AckermannDriveStamped}(drive_topic, queue_size = 10) # Publish to drive

    set_params!(c, [1., 1.], [0.01, 2.], [0.1, 0.1]);
    x_ref = [x_list; x_list[end]*ones(c.H)]
    y_ref = [y_list; y_list[end]*ones(c.H)]

    # Warm up
    println("Warm up")
    set_refs!(c, x_ref[1:c.H], y_ref[1:c.H]);
    start = to_sec(get_rostime())
    v, de = control(c)
    set_inputs!(c, [0.; 0.])
    t_hist = to_sec(get_rostime())- start
    println(t_hist)

    # Need a quick sleep here
    sleep(Duration(1.0))
    # Start the main loop
    loop_rate = Rate(5.0)
    i = 1
    while ! is_shutdown()
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = get_rostime()
        if (i <= L) & (c.start == 1)
            println("Step i = $i")
            set_refs!(c, x_ref[i:i+c.H-1], y_ref[i:i+c.H-1]);
            start = to_sec(get_rostime())
            v, de = control(c)
            t_hist = to_sec(get_rostime())- start
            println(t_hist)
            set_inputs!(c, [v, de])

            drive_msg.drive.speed = v
            drive_msg.drive.steering_angle = de
            publish(drive_pub, drive_msg)
            println(c.start)
            i += 1
        else
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
        end
        publish(drive_pub, drive_msg)
        rossleep(loop_rate)
    end
end

if ! isinteractive()
    main()
end
