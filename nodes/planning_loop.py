#!/usr/bin/env python3

# This is the main loop for planning running at approx 50Hz (though it is not doing much at the moment).

import rospy
import math
import tf
from threading import Lock
from car_demo.msg import LaneCoefficients, Trajectory
import copy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
import std_msgs.msg


class PlanningInputCollector:
    def __init__(self):
        """
        The class constructor.
        """
        self.__mutex = Lock()
        self.__coeffs = LaneCoefficients()
        self.__isNew = False
        rospy.Subscriber(
            "lane_coeffs", LaneCoefficients, self.callbackLaneCoeffs, queue_size=1
        )

    def callbackLaneCoeffs(self, msg):
        self.__mutex.acquire()
        self.__coeffs = msg
        self.__isNew = True
        self.__mutex.release()

    def haveNew(self):
        self.__mutex.acquire()
        isNew = copy.copy(self.__isNew)
        self.__mutex.release()
        return isNew

    def getNew(self):
        if self.haveNew():
            self.__mutex.acquire()
            coeffs = copy.copy(self.__coeffs)
            self.__isNew = False
            self.__mutex.release()
            return coeffs
        else:
            return None


def LS_lane_compute_center_3d(Z, maxDist=60, step=0.5, minDist=0):
    """
    Compute lane center coords from lane coeffs.
    """
    x_pred = np.arange(minDist, maxDist, step)
    offset = 0
    N = x_pred.shape[0]

    H = np.zeros((N, 4))  # design matrix
    u = x_pred.reshape((N,))
    H[:, 0] = offset * np.ones((N,))
    H[:, 1] = -np.ones((N,))
    H[:, 2] = -u
    H[:, 3] = 0.5 * np.multiply(u, u)

    y_pred = np.dot(H, Z)
    return (x_pred, y_pred)


def computeTrajectoryWorld(coeffs, trans, rpy):
    """
    Create a trajectory from lane coefficients and with desired speed of 15km/h.
    """
    xc_pred, yc_pred = LS_lane_compute_center_3d(
        np.matrix([coeffs.W, coeffs.Y_offset, coeffs.dPhi, coeffs.c0]).reshape((4, 1)),
        maxDist=15,
        step=0.25,
    )

    trajectory = Trajectory()
    trajectory.header = std_msgs.msg.Header()
    trajectory.header.stamp = coeffs.header.stamp  # time stamp of input data
    trajectory.header.frame_id = "map"

    T = np.array(
        [
            [math.cos(rpy[2]), -math.sin(rpy[2]), trans[0]],
            [math.sin(rpy[2]), math.cos(rpy[2]), trans[1]],
        ]
    )
    xc_pred = xc_pred.reshape((-1, 1))
    yc_pred = yc_pred.reshape((-1, 1))
    pts = np.hstack((xc_pred, yc_pred, np.ones_like(xc_pred)))
    pts_world = np.dot(T, pts.transpose())
    dpts_world = np.diff(pts_world, 1)

    theta = np.ravel(np.arctan2(dpts_world[1, :], dpts_world[0, :]))
    c = coeffs.c0 * np.ones_like(theta)

    v = 15 / 3.6 * np.ones_like(theta)
    s = np.zeros_like(v)
    temp = np.ravel(np.cumsum(np.sqrt(np.sum(np.multiply(dpts_world, dpts_world), 0))))
    s[1:] = temp[:-1]

    trajectory.x = np.ravel(pts_world[0, :-1]).tolist()
    trajectory.y = np.ravel(pts_world[1, :-1]).tolist()
    trajectory.theta = theta.tolist()
    trajectory.c = c.tolist()
    trajectory.v = v.tolist()
    trajectory.s = s.tolist()
    return trajectory, pts_world


if __name__ == "__main__":
    rospy.init_node("planning_node", log_level=rospy.DEBUG)

    display = False
    if display:
        display_cnt = 0
        rospy.logwarn(
            "Using display in planning loop is terribly slow! Try rqt_plot instead."
        )

    inputCollector = PlanningInputCollector()
    listener = tf.TransformListener()
    pubPath = rospy.Publisher("path_world", Path, queue_size=1)
    pubTrajectory = rospy.Publisher("trajectory", Trajectory, queue_size=1)

    rate = rospy.Rate(50.0)
    while not rospy.is_shutdown():
        if inputCollector.haveNew():
            coeffs = inputCollector.getNew()
            rospy.logdebug("Received new coefficients")

            if listener.canTransform("/map", "/din70000", coeffs.header.stamp):
                try:
                    (trans, rot) = listener.lookupTransform(
                        "/map", "/din70000", coeffs.header.stamp
                    )
                    rpy = tf.transformations.euler_from_quaternion(rot)
                except (
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                ):
                    continue

                rospy.logdebug("T = " + str(trans))
                rospy.logdebug("rpy = " + str(rpy))

                # compute path based on lane recognition
                trajectory, pts_world = computeTrajectoryWorld(coeffs, trans, rpy)

                # publish path
                pubTrajectory.publish(trajectory)

                path = Path()
                path.header = std_msgs.msg.Header()
                path.header.stamp = coeffs.header.stamp  # time stamp of input data
                path.header.frame_id = "map"
                path.poses = []
                for i in range(pts_world.shape[1]):
                    stampedPose = PoseStamped()
                    stampedPose.header = path.header
                    stampedPose.pose = Pose(
                        position=Point(x=pts_world[0, i], y=pts_world[1, i], z=0),
                        orientation=Quaternion(0, 0, 0, 0),
                    )  # omit orientation here
                    # since we only show points
                    path.poses.append(stampedPose)
                pubPath.publish(path)

            else:
                rospy.logdebug(
                    "Could not retrieve transform at time: %i.%i",
                    coeffs.header.stamp.secs,
                    coeffs.header.stamp.nsecs,
                )

        rate.sleep()
