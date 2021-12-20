#!/usr/bin/env python3

# This is a sample node that publishes arbitrary lane coefficients

import rospy
import sys
import numpy as np
import cv2
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from car_demo.msg import Control, LaneCoefficients
from simulation_image_helper import SimulationImageHelper


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
        self.Z_opt = np.array([4, 0, 0, 0.04]).T

    def callback(self, message):
        # get image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(message, "mono8")
        except CvBridgeError as e:
            rospy.logerr("Error in ControlTestLoop, callback: %s", e)
            return








        # img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(cv_image, (3,3), 0)

        ratio = 4
        kernelSize = 3
        threshold1 = 50
        threshold2 = threshold1*ratio

        # Canny Edge Detection
        edges = cv2.Canny(img_blur, threshold1, threshold2, kernelSize) # Canny Edge Detection
        # mask = edges != 0
        # newImg = cv_image * (mask[:,:,None].astype(cv_image.dtype))


        indices = np.where(edges != [0])
        M = np.column_stack((indices[0], indices[1]))
        # rospy.loginfo("M:")
        # rospy.loginfo(M)
        max_range_m = 45
        roi_left_line = np.array([
            [3, 0],
            [3, 4],
            [8, 4],
            [8, -4],
            [4, 0] ])
        roi_right_line = np.array([
            [3, 0],
            [3, -4],
            [8, -4],
            [8, 4],
            [4, 0] ])

        lane_left = np.empty((0,2))
        lane_right = np.empty((0,2))

        # for i in range(M.shape[0]):
        #     if cv2.pointPolygonTest(roi_left_line, (M[i,0], M[i,1]), False) > 0:
        #         lane_left = np.vstack((lane_left, M[i,:])) 
        #     if cv2.pointPolygonTest(roi_right_line, (M[i,0], M[i,1]), False) > 0:
        #         lane_right = np.vstack((lane_right, M[i,:]))

        cv_image_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        roi_left_line_transform = self.imageHelper.road2image(roi_left_line)
        roi_right_line_transform = self.imageHelper.road2image(roi_right_line)
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

        self.pubCannyImage.publish(self.bridge.cv2_to_imgmsg(cv_image_color))









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


def main(args):
    rospy.init_node("lane_detection_loop")
    myLoop = ControlTestLoop()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
