import matplotlib.pyplot as plt
import cvxpy
import numpy as np
import math
from BCI.dataProcessor import EEGDataProcessor
from BCI.ReceiveData import LSLDataCollector
from utils.Road import TrackGenerator
import time as ti

show_animation = True
# ==== Global Parameters ====
# State and input dimensions
NX = 4  # [x, y, v, yaw]
NU = 2  # [acceleration, steering]

# MPC horizon and cost matrices
Np = 5  # prediction horizon
R = np.diag([0.1, 0.01])       # input cost
Rd = R
Q = np.diag([1.0, 1.0, 1.0, 0.01])  # state cost
Qf = Q                         # terminal cost

# Goal conditions
GOAL_DIS = 0.2                # goal distance threshold [m]
MAX_ITER = 1

# Time parameters
DT = 0.2                      # [s] control time step
DT_bci = 0.5                  # [s] BCI control interval
N_IND_SEARCH = 10            # number of search indices

# Vehicle limits
MAX_STEER = np.deg2rad(25)    # max steering angle [rad]
MAX_DSTEER = np.deg2rad(18)   # max steering change per step [rad/s]
MAX_SPEED = 3.6 / 3.6         # [m/s]
MIN_SPEED = -3.6 / 3.6        # [m/s]
MAX_ACCEL = 1.0               # [m/s^2]

# Vehicle parameters
WB = 0.3
LENGTH = 0.5
WIDTH = 0.25
BACKTOWHEEL = 0.1
WHEEL_LEN = 0.1
WHEEL_WIDTH = 0.05
TREAD = 0.15

# Turn radius and angular speed
r = 5                        # turning radius [m]
vr = 3.6                     # target speed [m/s]

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Normalize an angle to the range (-π, π] or [0, 2π), optionally in degrees.

    Parameters:
        x (float or array-like): Angle(s) to be normalized.
        zero_2_2pi (bool): If True, normalize to [0, 2π); else to (-π, π].
        degree (bool): If True, input and output are in degrees.

    Returns:
        float or np.ndarray: Normalized angle(s).
    """
    is_float = isinstance(x, float)
    x = np.asarray(x).flatten()

    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    return mod_angle.item() if is_float else mod_angle

def pi_2_pi(angle):
    """
    Wrap angle to the range (-π, π].

    Parameters:
        angle (float or array-like): Angle(s) to wrap.

    Returns:
        float or np.ndarray: Wrapped angle(s).
    """
    return angle_mod(angle)

class State:
    """
    Vehicle state class.

    Attributes:
        x (float): x-coordinate of the vehicle.
        y (float): y-coordinate of the vehicle.
        yaw (float): heading angle (rad).
        v (float): velocity of the vehicle.
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

def update_state(state, a, delta):
    """
    Update the vehicle state using kinematic bicycle model.

    Parameters:
        state (State): Current state of the vehicle.
        a (float): Acceleration input.
        delta (float): Steering angle input.

    Returns:
        State: Updated vehicle state.
    """
    # Limit steering angle to within bounds
    delta = max(min(delta, MAX_STEER), -MAX_STEER)

    # Update state based on motion model
    state.x += state.v * math.cos(state.yaw) * DT
    state.y += state.v * math.sin(state.yaw) * DT
    state.yaw += state.v / WB * math.tan(delta) * DT
    state.v += a * DT

    # Clip velocity to min/max bounds
    state.v = max(min(state.v, MAX_SPEED), MIN_SPEED)

    return state

def update_state_bci(state, a, delta):
    """
    Update vehicle state for BCI scenario using different time step.

    Parameters:
        state (State): Current state of the vehicle.
        a (float): Acceleration input.
        delta (float): Steering angle input.

    Returns:
        State: Updated vehicle state.
    """
    # Limit steering angle
    delta = max(min(delta, MAX_STEER), -MAX_STEER)

    # Update with BCI time step
    state.x += state.v * math.cos(state.yaw) * DT_bci
    state.y += state.v * math.sin(state.yaw) * DT_bci
    state.yaw += state.v / WB * math.tan(delta) * DT_bci
    state.v += a * DT_bci

    # Clip velocity
    state.v = max(min(state.v, MAX_SPEED), MIN_SPEED)

    return state

def get_nparray_from_matrix(x):
    """
    Convert matrix-like input to a flattened NumPy array.

    Parameters:
        x (array-like): Input matrix.

    Returns:
        np.ndarray: Flattened array.
    """
    return np.array(x).flatten()

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    """
    Generate reference trajectory over the prediction horizon.

    Parameters:
        state (State): Current vehicle state.
        cx, cy (list): Path x and y coordinates.
        cyaw (list): Yaw angles of the path.
        ck (list): Curvatures (unused here).
        sp (list): Speed profile.
        dl (float): Distance interval between path points.
        pind (int): Previous closest index to ensure forward progression.

    Returns:
        tuple:
            diff_v (float): Speed difference from reference.
            diff_yaw (float): Yaw angle difference from reference.
            xref (np.ndarray): Reference trajectory [4 x (T+1)].
            ind (int): Updated closest index.
            dref (np.ndarray): Steering angle reference [1 x (T+1)].
    """
    xref = np.zeros((NX, Np + 1))
    dref = np.zeros((1, Np + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    # Ensure index doesn't move backwards
    if pind >= ind:
        ind = pind

    # Set initial reference point
    xref[:, 0] = [cx[ind], cy[ind], sp[ind], cyaw[ind]]
    dref[0, 0] = 0.0

    # Compute difference between current and reference state
    diff_v = state.v - sp[ind] / 3.6  # Convert to m/s if needed
    diff_yaw = state.yaw - cyaw[ind]

    # Wrap yaw difference to [-pi, pi]
    if diff_yaw > np.pi:
        diff_yaw -= 2 * np.pi
    elif diff_yaw < -np.pi:
        diff_yaw += 2 * np.pi

    travel = 0.0

    # Generate prediction horizon reference
    for i in range(1, Np + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if ind + dind < ncourse:
            xref[:, i] = [cx[ind + dind], cy[ind + dind],
                          sp[ind + dind], cyaw[ind + dind]]
        else:
            # Use last point if beyond path
            xref[:, i] = [cx[-1], cy[-1], sp[-1], cyaw[-1]]

        dref[0, i] = 0.0  # Reference delta (assumed to be 0)

    return diff_v, diff_yaw, xref, ind, dref

def calc_nearest_index(state, cx, cy, cyaw, pind):
    """
    Calculate the nearest index of the path to the current state.
    """
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]  # Squared Euclidean distances

    mind = min(d)
    ind = d.index(mind) + pind  # Adjust index relative to full path
    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))  # Directional angle error

    if angle < 0:
        mind *= -1

    return ind, mind

def predict_motion(x0, oa, od, xref):
    """
    Predict vehicle motion over time given initial state and control inputs.
    """
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for i, (ai, di) in enumerate(zip(oa, od), start=1):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    Iterative linear MPC control with updated linearization points.
    """
    ox = oy = oyaw = ov = None

    if oa is None or od is None:
        oa = [0.0] * Np  # Initial acceleration inputs
        od = [0.0] * Np  # Initial steering inputs

    for _ in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]

        cost, oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa[i] - poa[i]) for i in range(Np)) + \
             sum(abs(od[i] - pod[i]) for i in range(Np))

        if du <= 0.1:
            break

    return cost, oa, od, ox, oy, oyaw, ov

def calculate_optimization_cost_and_constraints(Np, u, x, xref, xbar, dref, R, Q, Rd, Qf,
                                                x0, MAX_SPEED, MIN_SPEED, MAX_ACCEL, MAX_STEER, MAX_DSTEER, DT):
    """
    Construct MPC optimization cost function and constraints.
    """
    cost = 0
    constraints = []

    for t in range(Np):
        # Control input cost
        cost += cvxpy.quad_form(u[:, t], R)

        # Deviation from reference state
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        # Linear model dynamics
        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        # Input rate change cost and constraint
        if t < (Np - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    # Terminal cost
    cost += cvxpy.quad_form(xref[:, Np] - x[:, Np], Qf)

    # Initial condition constraint
    constraints += [x[:, 0] == x0]

    # Speed constraints
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]

    # Acceleration and steering constraints
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    return cost, constraints

def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    x = cvxpy.Variable((NX, Np + 1))
    u = cvxpy.Variable((NU, Np))
    cost, constraints = calculate_optimization_cost_and_constraints(Np, u, x, xref, xbar, dref, R, Q, Rd, Qf,
                                                                    x0, MAX_SPEED, MIN_SPEED, MAX_ACCEL, MAX_STEER,
                                                                    MAX_DSTEER, DT)
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    #prob.solve(solver=cvxpy.ECOS, verbose=False)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return cost, oa, odelta, ox, oy, oyaw, ov

def check_goal(self, state, goal, target_ind):
    """
    Check if the vehicle has reached the goal.

    Args:
        state: current vehicle state
        goal: final goal position [x, y]
        target_ind: target index on the trajectory

    Returns:
        is_goal: True if goal reached
        distance: Euclidean distance to goal
    """
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    distance = math.hypot(dx, dy)

    is_goal = (distance <= GOAL_DIS)

    # Ensure sufficient progress along the path
    MIN_PROGRESS_PERCENTAGE = 0.9
    is_enough_progress = (target_ind / self.total_point) >= MIN_PROGRESS_PERCENTAGE

    if is_goal and is_enough_progress:
        return True, distance
    return False, distance

def smooth_yaw(yaw):
    """
    Smooth yaw angles to avoid discontinuities (e.g., 180° to -180°).

    Args:
        yaw: list of yaw angles

    Returns:
        yaw: smoothed yaw list
    """
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def calc_ed_ephi(x, y, phi, xr, yr, thetar, kappar):
    """
    Calculate lateral error (ed) and heading error (ephi).

    Args:
        x, y: current position
        phi: current heading
        xr, yr: reference trajectory
        thetar: reference yaw angles
        kappar: curvature

    Returns:
        ed: lateral error
        ephi: heading error
    """
    n = len(xr)
    d_min = (x - xr[0]) ** 2 + (y - yr[0]) ** 2
    dmin = 0

    for i in range(n):
        d = (x - xr[i]) ** 2 + (y - yr[i]) ** 2
        if d < d_min:
            d_min = d
            dmin = i

    # Tangent and normal vectors
    tor = np.array([np.cos(thetar[dmin]), np.sin(thetar[dmin])])
    nor = np.array([-np.sin(thetar[dmin]), np.cos(thetar[dmin])])
    d_err = np.array([x - xr[dmin], y - yr[dmin]])

    ed = np.dot(nor, d_err)
    es = np.dot(tor, d_err)
    projection_theta = thetar[dmin] + kappar[dmin] * es
    ephi = np.sin(phi - projection_theta)

    return ed, ephi

def threshold_based_trigger(state, ed_thresh, ephi_thresh, ev_thresh):
    """
    Check if event-triggering conditions are met based on lateral, heading, and speed errors.
    Returns 1 if triggered, 0 otherwise.
    """
    ed, ephi = calc_ed_ephi(
        state.x, state.y, state.yaw,
        vehEnv.cx, vehEnv.cy, vehEnv.cyaw, vehEnv.ck
    )
    diff_v = state.v - vr

    if abs(ed) >= ed_thresh or abs(ephi) >= ephi_thresh or abs(diff_v) >= ev_thresh:
        return 1
    else:
        return 0

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k",zoom=0.0):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def do_simulation(self, index, time, cx, cy, cyaw, ck, sp, dl, initial_state, action, outputlabels):
    """
    Perform simulation step.

    Args:
        index: last target index
        time: current simulation time
        cx, cy: reference path x and y
        cyaw: reference yaw
        ck: reference curvature
        sp: speed profile
        dl: distance step
        initial_state: vehicle state at beginning of simulation
        action: controller type (0: brain control, 1: MPC)
        outputlabels: output command label from decoder

    Returns:
        is_done, target_ind, next_state, reward, done
    """
    goal = [cx[-1], cy[-1]]
    state = initial_state
    is_done = True

    # Fix yaw wraparound
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= 2.0 * math.pi
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += 2.0 * math.pi

    odelta, oa = None, None
    target_ind = index
    cyaw = smooth_yaw(cyaw)

    # Reference trajectory and initial state
    diff_v, diff_yaw, xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)
    ed, ephi = calc_ed_ephi(state.x, state.y, state.yaw, self.cx, self.cy, self.cyaw, self.ck)
    x0 = [state.x, state.y, state.v, state.yaw]

    self.start_time = ti.time()

    if action == 1:
        # MPC control
        self.k = 0
        cost, oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)
        self.elapsed_time = ti.time() - self.start_time
        self.total_time += self.elapsed_time

        self.oa, self.odelta = oa, odelta
        cost = cost.value
        if odelta is not None:
            self.ai, self.di = oa[0], odelta[0]
            state = update_state(state, self.ai, self.di)
            ed, ephi = calc_ed_ephi(state.x, state.y, state.yaw, self.cx, self.cy, self.cyaw, self.ck)
        time += DT
        outputlabels = 0
    else:
        # Brain control using outputlabels
        if outputlabels == 13:
            di, self.ai = -np.deg2rad(self.bdelta), 0
        elif outputlabels == 12:
            di, self.ai = np.deg2rad(self.bdelta), 0
        elif outputlabels == 11:
            self.di, self.ai = 0, self.ai + self.adelta
        elif outputlabels == 10:
            self.di, self.ai = 0, self.ai - self.adelta
        else:
            self.di, self.ai = 0, 0

        state = update_state_bci(state, self.ai, self.di)

        ed, ephi = calc_ed_ephi(state.x, state.y, state.yaw, self.cx, self.cy, self.cyaw, self.ck)
        time += DT_bci

    # Check if goal is reached
    done, distance = check_goal(self, state, goal, target_ind)
    ed_thresh = 0.15
    ephi_thresh = 10 * np.pi / 180
    ev_thresh = 0.20
    reward = - ((np.abs(ed) / ed_thresh) +
                (np.abs(ephi) / ephi_thresh) +
                (np.abs(diff_v) / ev_thresh)) - self.rho * action

    next_state = state

    # Handle termination
    if done:
        is_done = True
    if time >= self.MAXTIME:
        done = True
        is_done = False
        print("timeout")

    if show_animation:
        #zoom = 5
        plt.cla()
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        plot_car(state.x, state.y, state.yaw, steer=self.di)
        plt.plot(self.x_left, self.y_left)
        plt.plot(self.x_right, self.y_right)
        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2))
                  + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
        # plt.xlim(state.x - zoom, state.x + zoom)
        # plt.ylim(state.y - zoom, state.y + zoom)
        plt.pause(0.001)

    return is_done, target_ind, next_state, reward, done

class env_cars:
    def __init__(self, dl=0.1):
        """
               Initialize the environment for a driverless car using MPC and BCI control.
               :param dl: Path discretization step [m]
               :param vr: Target velocity [m/s]
               """
        self.dl = dl
        self.vr = vr

        # Generate track and associated parameters
        self.road = TrackGenerator(dl, target_speed=vr)
        self.cx, self.cy, self.cyaw, self.ck, self.sp, \
            self.x_left, self.x_right, self.y_left, self.y_right = self.road.generate_track()
        self.total_point = len(self.cx)
        self.time = 0

        # Action/state space definitions
        self.action_space = 2
        self.obs_space = 4

        # Simulation states
        self.iniState = State
        self.firstindex = 0
        self.k = 0

        # Logging variables
        self.oa = np.zeros(Np)
        self.odelta = np.zeros(Np)
        self.ox = np.zeros(Np + 1)
        self.oy = np.zeros(Np + 1)
        self.ov = np.zeros(Np + 1)
        self.oyaw = np.zeros(Np + 1)

        # Control inputs
        self.di = 0
        self.ai = 0
        self.bdi = 0

        # MPC parameters
        self.rho = 0.1
        self.DT = DT

        # Vehicle dynamics
        self.r = r
        self.omega = vr / r
        self.delta_theta_deg = math.degrees(self.omega)

        # Bci Control parameters
        self.bdelta = 4
        self.adelta = 0.1

        # Vehicle limits
        self.MAX_STEER = MAX_STEER
        self.MAX_DSTEER = MAX_DSTEER
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.MAX_ACCEL = MAX_ACCEL

        # Brain control parameters
        self.total_time = 0
        self.start_time = 0

        #EEG data processor and collector initialization
        self.DataCollector = LSLDataCollector()
        self.DataCollector.initialize_inlet()
        self.DataProcessor = EEGDataProcessor()

        # Simulation time limit
        self.MAXTIME = 90

    def reset(self, yaw=np.pi / 2, v=1.8/3.6):
        """
        Reset the environment to the initial state.
        :param yaw: Initial yaw angle [rad]
        :param v: Initial speed [m/s]
        :return: Initial state
        """
        self.iniState = State(x=self.cx[0], y=self.cy[0], yaw=yaw, v=v)
        self.firstindex = 0
        self.time = 0
        return self.iniState

    def step(self, action,state):
        """
        Execute one step of simulation given the current action and state.
        :param action state

        :return: updated simulation results
        """
        self.DataCollector.collect_data()
        output_label = self.DataProcessor.predict_online(self.DataCollector.buffer)
        #print(output_label)

        result = do_simulation(
            self, self.firstindex, self.time, self.cx, self.cy,
            self.cyaw, self.ck, self.sp, self.dl, state,
            action, output_label
        )


        return is_done, next_state, reward, done

if __name__ == "__main__":
    # Initialize environment
    vehEnv = env_cars()
    state = vehEnv.reset()
    done = False

    # Initialize storage for results
    act_list = []

    while not done:
        # Get decision from threshold-based controller
        action = threshold_based_trigger(state, ed_thresh=0.15, ephi_thresh=10*np.pi/180, ev_thresh=0.20)
        print(action)

        # Record action and error velocity
        act_list.append(action)

        # Step the environment with action=1 (event triggered by MPC)
        is_done, next_state, reward, done = vehEnv.step(action, state)

        # Prepare for next step
        state = next_state

    # Output summary results
    print(f"Total accumulated time: {vehEnv.total_time:.4f} seconds")

    # Event triggering statistics
    trigger_count = act_list.count(1)
    total_actions = len(act_list)
    total_duration = total_actions * DT
    stimulus_freq = trigger_count / total_duration
    trigger_ratio = trigger_count / total_actions

    print(f"Trigger frequency per second: {stimulus_freq:.2f} Hz")
    print(f"Total trigger count: {trigger_count}")
    print(f"Total action count: {total_actions}")
    print(f"Trigger ratio: {trigger_ratio:.2%}")