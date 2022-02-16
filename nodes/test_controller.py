#!/usr/bin/env python3

# This is a sample node that publishes arbitrary lane coefficients

from numpy.linalg.linalg import LinAlgError
import rospy
import sys
import numpy as np
import cv2
import time
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from car_demo.msg import Control, LaneCoefficients
from simulation_image_helper import SimulationImageHelper
from m_estimator import LS_lane_residuals, LS_lane_inliers, Cauchy, MEstimator_lane_fit


class ControlTestLoop:
    def __init__(self):
        """
        setup all publishers and subscribers
        """
        self.pubLaneCoeffs = rospy.Publisher(
            "lane_coeffs", LaneCoefficients, queue_size=1
        )
        self.pubCannyImage = rospy.Publisher(
            "canny_dbg", Image, queue_size=1
        )
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/prius/front_camera/image_raw", Image, self.callback, queue_size=1
        )
        self.imageHelper = SimulationImageHelper()
        rospy.logdebug("Completed initialization")

        # a sample set of lane coefficients:
        #   W, Y_offset, dPhi, c0
        self.Z_opt = np.array([4, 0, 0, 0.0]).T
        self.Z_initial = np.array([4, 0, 0, 0]).T





    def callback(self, message):
        # get image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(message, "mono8")
        except CvBridgeError as e:
            rospy.logerr("Error in ControlTestLoop, callback: %s", e)
            return




        #________mask_____________________
        # h = cv_image.shape[1]
        # w = cv_image.shape[0]

        # mask = np.full_like(cv_image, 255)
        # mask = cv2.rectangle(mask, (int(w/10), h), (int((w/10)*9), h), (0,0,0), -1)
        # masked_image = np.bitwise_and(cv_image, mask)



        # img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(cv_image, (3,3), 0)

        ratio = 4
        kernelSize = 3
        threshold1 = 50
        threshold2 = threshold1*ratio

        # Canny Edge Detection
        edges = cv2.Canny(img_blur, threshold1, threshold2, kernelSize)  # Canny Edge Detection
        # mask = edges != 0
        # newImg = cv_image * (mask[:, :, None].astype(cv_image.dtype))

        # rospy.loginfo(type(newImg))

        # self.pubCannyImage.publish(self.bridge.cv2_to_imgmsg(self.bridge.cv2_to_imgmsg(cv_image)))


        indices = np.where(edges != [0])
        M = np.column_stack((indices[1], indices[0]))
        M = M[::3]
        # rospy.loginfo("M:")
        # rospy.loginfo(M)
        roi_left_line = np.array([
            [3, 0],  # mitte unten
            [3, 1.5],  # links unten
            [0, 1.5],
            [0, 2],
            [10, 4],  # links oben
            [10, -4],  # rechts oben
            [5, 0] ])  # mitte mitte
        roi_right_line = np.array([
            [3, 0],
            [3, -1.5],
            [0, -1.5],
            [0, -2],
            [10, -4],
            [10, 4],
            [5, 0] ])

        lane_left = np.empty((0,2))
        lane_right = np.empty((0,2))

        cv_image_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        roi_left_line_transform = self.imageHelper.road2image(roi_left_line)
        roi_right_line_transform = self.imageHelper.road2image(roi_right_line)

        # rospy.loginfo(roi_left_line_transform)
        # rospy.loginfo(roi_left_line_transform.astype(np.int32))
        # rospy.loginfo(type(roi_left_line_transform))

        for i in range(M.shape[0]):
            if cv2.pointPolygonTest(roi_left_line_transform.astype(np.int32), (M[i, 0], M[i, 1]), False) > 0:
                lane_left = np.vstack((lane_left, M[i, :])) 
            if cv2.pointPolygonTest(roi_right_line_transform.astype(np.int32), (M[i, 0], M[i, 1]), False) > 0:
                lane_right = np.vstack((lane_right, M[i, :]))

        lane_left_transform = self.imageHelper.image2road(lane_left)
        lane_right_transform = self.imageHelper.image2road(lane_right)

        cv2.polylines(
            cv_image_color,
            [roi_left_line_transform.astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=8,
        )
        cv2.polylines(
            cv_image_color,
            [roi_right_line_transform.astype(np.int32)],
            isClosed=True,
            color=(255, 0, 0),
            thickness=8,
        )
        # cv2.polylines(
        #     cv_image_color,
        #     [lane_left.astype(np.int32)],
        #     isClosed=True,
        #     color=(0, 255, 0),
        #     thickness=8,
        # )
        # cv2.polylines(
        #     cv_image_color,
        #     [lane_right.astype(np.int32)],
        #     isClosed=True,
        #     color=(0, 255, 255),
        #     thickness=8,
        # )
        for point_left in lane_left.astype(np.int32):
            cv2.circle(cv_image_color, tuple(point_left), 1, (0, 255, 0), 2)
        for point_right in lane_right.astype(np.int32):
            cv2.circle(cv_image_color, tuple(point_right), 1, (0, 255, 255), 2)

        self.pubCannyImage.publish(self.bridge.cv2_to_imgmsg(cv_image_color))

        try:
            start = time.time()
            Z_MEst = MEstimator_lane_fit(lane_left_transform, lane_right_transform, self.Z_initial, sigma=0.2, maxIteration=10)
            end = time.time()
            rospy.loginfo("Duration: " + str(end - start))
            rospy.loginfo(Z_MEst)
            self.Z_opt = Z_MEst
            self.Z_initial = Z_MEst
            self.Z_initial[0] = 4
        except LinAlgError as e:
            rospy.logerr_throttle(1, e)







        # send lane coeffs
        coeffs = LaneCoefficients()
        coeffs.header = std_msgs.msg.Header()
        coeffs.header.stamp = message.header.stamp  # time stamp of input data
        coeffs.header.frame_id = "din70000"
        coeffs.W = self.Z_opt[0]
        coeffs.Y_offset = self.Z_opt[1]
        coeffs.dPhi = self.Z_opt[2]
        coeffs.c0 = self.Z_opt[3]

        self.pubLaneCoeffs.publish(coeffs)
        rospy.logdebug(
            "published lane coefficients at stamp %i %i",
            coeffs.header.stamp.secs,
            coeffs.header.stamp.nsecs,
        )
        # rospy.spinOnce()


def main(args):
    rospy.init_node("lane_detection_loop")
    myLoop = ControlTestLoop()
    try:
        rospy.spin()
        # rospy.spinOnce()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
