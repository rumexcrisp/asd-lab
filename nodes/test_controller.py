#!/usr/bin/env python3

# This is a sample node that publishes arbitrary lane coefficients

import cv_bridge
import rospy
import sys
import numpy as np
from rospy.core import loginfo
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from car_demo.msg import Control, LaneCoefficients
from simulation_image_helper import SimulationImageHelper
import cv2
from matplotlib import pyplot as plt
from m_estimator import LS_lane_residuals, LS_lane_inliers, Cauchy, MEstimator_lane_fit, LS_lane_compute



class ControlTestLoop:
    def __init__(self):
        """
        setup all publishers and subscribers
        """
        self.imageHelper = SimulationImageHelper()
        
        self.pubLaneCoeffs = rospy.Publisher(
            "lane_coeffs", LaneCoefficients, queue_size=1
        )
        self.pubCannyImg = rospy.Publisher(
            "canny_dbg", Image, queue_size=1
        )
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/prius/front_camera/image_raw", Image, self.callback, queue_size=1
        )
        rospy.logdebug("Completed initialization")

        # a sample set of lane coefficients:
        #   W, Y_offset, dPhi, c0
        self.Z_opt = np.array([4, 0, 0, 0.0]).T
        self.Z_MEst = self.Z_opt


    def callback(self, message):
        # get image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(message, "mono8")
        except CvBridgeError as e:
            rospy.logerr("Error in ControlTestLoop, callback: %s", e)
            return

        #_____________Calc images__________________________#
        threshold = 50
        ratio = 4
        kernel_size = 3
        im_blur = cv2.GaussianBlur(cv_image, (3,3), 0)
        edges = cv2.Canny(im_blur, threshold, threshold*ratio, kernel_size)

        #____________

        indices = np.where(edges != [0])
        M = np.column_stack((indices[1], indices[0]))
        #_______________________________________________________

        roi_left_line = np.array([
            [4, 0.2], 
            [4, 3], 
            [8, 3], 
            [8, 1],
            [4, 0.2]])

        roi_right_line = np.array([
            [4, -0.2], 
            [4, -3], 
            [8, -3], 
            [8, -1],
            [4, -0.2]])

        lane_left = np.empty((0,2))
        lane_right = np.empty((0,2))

        roi_right_img = self.imageHelper.road2image(roi_right_line)
        roi_left_img = self.imageHelper.road2image(roi_left_line)

        for i in range(M.shape[0]):
            if cv2.pointPolygonTest(roi_left_img.astype(np.int32), (M[i,0], M[i,1]), False) > 0:
                lane_left = np.vstack((lane_left, M[i,:])) 
            if cv2.pointPolygonTest(roi_right_img.astype(np.int32), (M[i,0], M[i,1]), False) > 0:
                lane_right = np.vstack((lane_right, M[i,:]))
        

        cv_image_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


        cv2.polylines(
            cv_image_color,
            [roi_right_img.astype(np.int32)],
            isClosed=True,
            color=(100, 0, 0),
            thickness=3,
        )
        cv2.polylines(
            cv_image_color,
            [roi_left_img.astype(np.int32)],
            isClosed=True,
            color=(0, 100, 0),
            thickness=3,
        )

        

        #____________calc lane coeffs______________________#
        

        ll = self.imageHelper.image2road(lane_left)
        lr = self.imageHelper.image2road(lane_right)

        try:
            self.Z_MEst = MEstimator_lane_fit(ll, lr, self.Z_MEst, sigma=0.1, maxIteration=10)
        except np.linalg.LinAlgError:
            self.Z_MEst = self.Z_opt

        for p in lane_left:
            cv2.circle(cv_image_color, tuple(p.astype(np.int32)), 2, (0, 0, 255))
        
        for p in lane_right:
            cv2.circle(cv_image_color, tuple(p.astype(np.int32)), 2, (255, 0, 0))

        xpred, ypredL, ypredR = LS_lane_compute(self.Z_MEst)


        lp = np.column_stack((xpred, ypredL))
        lp = self.imageHelper.image2road(lp)

        for l in lp:
            rospy.loginfo(l[0][])
            # cv2.circle(cv_image_color, tuple(l.astype(np.int32)), 2, (0, 0, 255))
        
        # for i in range(ypredR.astype(np.int32)):
        #     cv2.circle(cv_image_color, tuple(xpred[i], ypredR[i]), 2, (255, 0, 0))

        img_edges = self.bridge.cv2_to_imgmsg(cv_image_color)
        self.pubCannyImg.publish(img_edges)


        # send lane coeffs
        coeffs = LaneCoefficients()
        coeffs.header = std_msgs.msg.Header()
        coeffs.header.stamp = message.header.stamp  # time stamp of input data
        coeffs.header.frame_id = "din70000"
        coeffs.W = self.Z_MEst[0]
        coeffs.Y_offset = self.Z_MEst[1]
        coeffs.dPhi = self.Z_MEst[2]
        coeffs.c0 = self.Z_MEst[3]

        self.pubLaneCoeffs.publish(coeffs)
        rospy.logdebug(
            "published lane coefficients at stamp %i %i",
            coeffs.header.stamp.secs,
            coeffs.header.stamp.nsecs,
        )


def main(args):
    rospy.init_node("lane_detection_loop")
    myLoop = ControlTestLoop()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
