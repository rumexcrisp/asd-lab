#!/usr/bin/env python3

# This is the main loop for planning

import rospy
import tf
from threading import Lock
from car_demo.msg import Control, Trajectory, ControlDebug
import copy
import numpy as np
from sensor_msgs.msg import JointState
import message_filters
import std_msgs.msg


class VehicleParams:
    """
    Class representing all vehicle and simulation parameters.

    dT [= 0.1]:                         [s] simulation time step
    L [= 2.9]:                          [m] Wheel base of vehicle
    veh_dim_x, veh_dim_y [= 4, 1.9]:    [m] size of vehicle (length, width)
    max_steer [= np.radians(30.0)]:     [rad] max steering angle
    max_ax [= 2]:                       [m/ss] max (positive) acceleration
    min_ax [= -10]:                     [m/ss] max deceleration (=min negative acceleration)
    """

    def __init__(
        self,
        dT=0.1,
        L=2.9,
        veh_dim_x=4,
        veh_dim_y=1.9,
        max_steer=np.radians(30.0),
        max_ax=2,
        min_ax=-10,
    ):
        """Instantiate the object."""
        self.dT = dT  # [s] simulation time step
        self.L = L  # [m] Wheel base of vehicle
        self.veh_dim_x, self.veh_dim_y = (
            veh_dim_x,
            veh_dim_y,
        )  # [m] size of vehicle (length, width)
        self.max_steer = max_steer  # [rad] max steering angle
        self.max_ax = max_ax  # [m/ss] max (positive) acceleration
        self.min_ax = min_ax  # [m/ss] max deceleration (=min negative acceleration)


class State:
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calc_target_index(state, cx, cy, cyaw, params=VehicleParams()):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [m] x-coordinates of (sampled) desired trajectory
    :param cy: [m] y-coordinates of (sampled) desired trajectory
    :param cyaw: [rad] tangent angle of (sampled) desired trajectory
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + 0.5 * params.L * np.cos(state.yaw)
    fy = state.y + 0.5 * params.L * np.sin(state.yaw)

    # Search nearest point index
    dx_vec = fx - np.asarray(cx).reshape([-1, 1])
    dy_vec = fy - np.asarray(cy).reshape([-1, 1])
    dist = np.hstack([dx_vec, dy_vec])
    dist_2 = np.sum(dist ** 2, axis=1)
    target_idx = np.argmin(dist_2)

    # Project RMS error onto front axle vector
    front_axle_vec = [
        np.cos(cyaw[target_idx] + np.pi / 2),
        np.sin(cyaw[target_idx] + np.pi / 2),
    ]
    error_front_axle = np.dot(dist[target_idx, :], front_axle_vec)

    return target_idx, error_front_axle


def stanley_control(
    state, cx, cy, cyaw, last_target_idx=0, k=0.7, params=VehicleParams()
):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: [m] x-coordinates of (sampled) desired trajectory
    :param cy: [m] y-coordinates of (sampled) desired trajectory
    :param cyaw: [rad] orientation of (sampled) desired trajectory
    :param last_target_idx: [int] last visited point on desired trajectory (set 0 if not available)
    :param k: control gain
    :return: ([rad] steering angle,
        [int] last visited point on desired trajectory,
        [m] cross track error at front axle)
    """
    current_target_idx, error_front_axle = calc_target_index(
        state, cx, cy, cyaw, params
    )

    # make sure that we never match a point on the desired path
    # that we already passed earlier:
    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(-k * error_front_axle, max(0.2 + state.v, 0.1))
    # Steering control
    print(
        "yaw (deg): %.3f, cyaw (deg): %.3f"
        % (state.yaw * 180.0 / np.pi, cyaw[current_target_idx] * 180.0 / np.pi)
    )
    print(
        "heading error (deg): %.3f, crosstrack error (m): %.3f"
        % (theta_e * 180.0 / np.pi, theta_d)
    )
    delta = theta_e + theta_d

    return delta, current_target_idx, error_front_axle, theta_e


def speed_control(target, current, Kp=1.0):
    """
    Proportional control for the speed.
    :param target: target speed (m/s)
    :param current: current speed (m/s)
    :param Kp: speed proportional gain
    :return: controller output (m/ss)
    """
    return Kp * (target - current)


class ControlInputCollector:
    def __init__(self):
        """
        The class constructor.
        """
        self.__mutex = Lock()
        self.__traj = Trajectory()
        self.__isNew = False
        rospy.Subscriber(
            "trajectory", Trajectory, self.callbackTrajectory, queue_size=1
        )

    def callbackTrajectory(self, msg):
        self.__mutex.acquire()
        self.__traj = msg
        self.__isNew = True
        self.__mutex.release()

    def haveNew(self):
        """
        This returns True if a new trajectory is available, that we haven't retrieved with
        "getLatest" yet. Once "getLatest" has been called, it's no longer new.
        """
        self.__mutex.acquire()
        isNew = copy.copy(self.__isNew)
        self.__mutex.release()
        return isNew

    def getLatest(self):
        self.__mutex.acquire()
        traj = copy.copy(self.__traj)
        self.__isNew = False
        self.__mutex.release()
        return traj


if __name__ == "__main__":
    rospy.init_node("control_node")

    inputCollector = ControlInputCollector()
    listener = tf.TransformListener()
    jointStateSub = message_filters.Subscriber("joint_states", JointState)
    jointStateCache = message_filters.Cache(jointStateSub, 100)

    pubControl = rospy.Publisher("prius", Control, queue_size=1)
    pubControlDebug = rospy.Publisher("ControlDebug", ControlDebug, queue_size=1)

    vehParams = VehicleParams(L=2.7, veh_dim_x=4.645, veh_dim_y=1.760)

    rate = rospy.Rate(50.0)
    dT = rospy.Duration(secs=0, nsecs=20000000)  # 20ms = 1/50Hz
    target_idx, traj, last_time = 0, None, rospy.Time()

    current_gear = None
    trans_old = None

    while not rospy.is_shutdown():
        if inputCollector.haveNew():
            target_idx = 0
            traj = inputCollector.getLatest()
            rospy.logdebug("Received new trajectory")

        if traj:
            new_time = listener.getLatestCommonTime("/map", "/din70000")

            if new_time - last_time > dT:
                rospy.logdebug(
                    "Checking new positions (%s -> %s)", str(last_time), str(new_time)
                )

                try:
                    (trans, rot) = listener.lookupTransform(
                        "/map", "/din70000", new_time
                    )
                    rpy = tf.transformations.euler_from_quaternion(rot)
                except (
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                ):
                    continue

                joint_state = jointStateCache.getElemBeforeTime(new_time)
                joint_state_dict = dict(zip(joint_state.name, joint_state.velocity))

                r_wheel = 0.31265
                velocity_kmh = (
                    0.25
                    * (
                        joint_state_dict["front_right_wheel_joint"]
                        + joint_state_dict["rear_left_wheel_joint"]
                        + joint_state_dict["front_left_wheel_joint"]
                        + joint_state_dict["rear_right_wheel_joint"]
                    )
                    * r_wheel
                    * 3.6
                )

                rospy.logdebug("T = " + str(trans))
                if trans_old is not None:
                    dtrans = np.asarray(trans) - np.asarray(trans_old)
                    v_kmh_ref = max(
                        0,
                        3.6 * np.linalg.norm(dtrans) / (new_time - last_time).to_sec(),
                    )
                    rospy.loginfo("v_kmh_ref = " + str(v_kmh_ref))
                else:
                    v_kmh_ref = velocity_kmh

                rospy.logdebug("rpy = " + str(rpy))
                rospy.loginfo("v_kmh = " + str(velocity_kmh))

                trans_old = trans
                last_time = new_time

                state = State(
                    trans[0],
                    trans[1],
                    normalize_angle(rpy[2]),
                    v_kmh_ref / 3.6,
                )

                di, target_idx, dlat, dtheta = stanley_control(
                    state,
                    traj.x,
                    traj.y,
                    traj.theta,
                    target_idx,
                    k=0.9,
                    params=vehParams,
                )
                di = di * 2

                ai = speed_control(traj.v[target_idx], state.v, Kp=0.2)

                command = Control()
                command.header = std_msgs.msg.Header()
                command.header.stamp = new_time
                # check if standing still, then keep applying brakes
                if (np.abs(traj.v[target_idx]) < 0.1) and (np.abs(state.v) < 0.5):
                    command.throttle, command.brake = 0, 0.5
                else:
                    if ai > 0:
                        command.throttle, command.brake = np.clip(ai, 0, 1), 0
                    else:
                        command.throttle, command.brake = 0, np.clip(-ai, 0, 1)
                command.steer = np.clip(di, -1, 1)

                if current_gear != Control.FORWARD:
                    current_gear = Control.FORWARD
                    command.shift_gears = current_gear

                pubControl.publish(command)

                controlDebug = ControlDebug()
                controlDebug.header = std_msgs.msg.Header()
                controlDebug.header.stamp = new_time

                controlDebug.x = state.x
                controlDebug.y = state.yaw
                controlDebug.s = traj.s[target_idx]
                controlDebug.theta = state.yaw
                controlDebug.v = state.v
                controlDebug.dlat = dlat
                controlDebug.dtheta = dtheta
                controlDebug.steering = di
                controlDebug.acceleration = ai
                pubControlDebug.publish(controlDebug)

                rospy.loginfo(
                    "di=%.4f, trgt_idx=%d, dlat=%.2f, dtheta=%.4fdeg, dv=%.2f"
                    % (
                        di,
                        target_idx,
                        dlat,
                        dtheta * 180.0 / np.pi,
                        traj.v[target_idx] - state.v,
                    )
                )

        rate.sleep()
