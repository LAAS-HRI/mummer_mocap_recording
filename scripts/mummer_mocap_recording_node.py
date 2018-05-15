#!/usr/bin/env python

import math
import tf
import numpy as np
import rospy
from mummer_mocap_recording.msg import or_pose_estimator_state
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
import cv2
from cv_bridge import CvBridge
from tf.transformations import translation_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_matrix, concatenate_matrices

CAMERA_DEPTH_FRAME = "CameraDepth_optical_frame"

CAMERA_SUB_TOPIC = "/naoqi_driver_node/camera/ir/image_raw/compressed"
CAMERA_PARAM_SUB_TOPIC = "/naoqi_driver_node/camera/depth/camera_info"

CHESSBOARD_IMAGE_FRAME = "chessboard_image_frame"
CHESSBOARD_RB_FRAME = "mocap_chessboard"
REFERENCE_FRAME = "head_ref"

chessboard_size = (9,6)
chessboard_square_size = 0.0246875

chessboard_image_2_rb_translation = (0, 0, 0)
chessboard_image_2_rb_rotation = (1,0,0,0)

def transformation_matrix(t, q):
    translation_mat = translation_matrix(t)
    rotation_mat = quaternion_matrix(q)
    return np.dot(translation_mat, rotation_mat)

class MummerMocapRecording(object):

    def __init__(self):
        self.person_0_pub = rospy.Publisher("/mummer_mocap_recording/person_0", PoseStamped, queue_size=5)
        self.chessboard_pub = rospy.Publisher("/mummer_mocap_recording/chessboard", PoseStamped, queue_size=5)
        self.head_ref_transform = None
        self.mocap_chessboard_transform = None
        self.image_chessboard_transform = None
        self.offset_transform = None
        self.person_0_pose = None
        self.bridge = CvBridge()
        self.image_np = None
        self.camera_matrix = None
        self.camera_distortion = None
        self.debug_im_pub = rospy.Publisher("/mummer_mocap_recording/debug", Image, queue_size=5)
        self.image_sub = rospy.Subscriber(CAMERA_SUB_TOPIC, CompressedImage, self._on_new_image)
        self.cam_param_sub = rospy.Subscriber(CAMERA_PARAM_SUB_TOPIC, CameraInfo, self._on_new_cam_info)
        self.mocap_chessboard = rospy.Subscriber("/optitrack/bodies/world", or_pose_estimator_state, self.callback_update_mocap_chessboard)
        self.person_0_sub = rospy.Subscriber("/optitrack/bodies/person_0", or_pose_estimator_state, self.callback_update_person_0)
        self.head_ref_sub = rospy.Subscriber("/optitrack/bodies/head_ref", or_pose_estimator_state, self.callback_update_head_ref)
        self._tf_broadcaster = tf.TransformBroadcaster()

        self.chessboard_model = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.chessboard_model[:, :2] = np.mgrid[0:chessboard_size[0] * chessboard_square_size:chessboard_square_size,
                                       0:chessboard_size[1] * chessboard_square_size:chessboard_square_size].T.reshape(
            -1, 2)

    def _on_new_image(self, ros_data):
        self.image_np = self.bridge.compressed_imgmsg_to_cv2(ros_data, "bgr8")

    def _on_new_cam_info(self, infos):
        self.camera_matrix = np.array(infos.P).reshape(3,4)[:, :3].reshape(3,3)
        self.camera_distortion = np.array(infos.D).reshape(5,1)

    def rodriguez_to_quaternion(self, r):
        """
        From http://ros-users.122217.n3.nabble.com/rotation-problem-on-published-tf-links-related-to-a-re-projected-opencv-checkerboard-in-rviz-td2058855.html
        :param r:
        :return:
        """
        mat, _ = cv2.Rodrigues(r)
        w = mat[0][0] + mat[1][1] + mat[2][2] + 1

        w = math.sqrt(w)
        quat = [(mat[2][1] - mat[1][2]) / (w * 2.0),
                (mat[0][2] - mat[2][0]) / (w * 2.0),
                (mat[1][0] - mat[0][1]) / (w * 2.0),
                w / 2.0]
        return quat

    def callback_update_mocap_chessboard(self, msg):
        if self.head_ref_transform is not None:
            if len(msg.pos)>0:
                self.mocap_chessboard_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z], [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz, msg.pos[0].qw])
                self.mocap_chessboard_transform = np.dot(np.linalg.inv(self.head_ref_transform), self.mocap_chessboard_transform)
                t = translation_from_matrix(self.mocap_chessboard_transform)
                q = quaternion_from_matrix(self.mocap_chessboard_transform)
                self._tf_broadcaster.sendTransform(t, q, rospy.Time.now(), CHESSBOARD_RB_FRAME, REFERENCE_FRAME)

    def callback_update_person_0(self, msg):
        if self.head_ref_transform is not None:
            if len(msg.pos)>0:
                person_0_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z], [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz, msg.pos[0].qw])
                person_0_transform = np.dot(np.linalg.inv(self.head_ref_transform), person_0_transform)
                # if self.offset_transform is not None:
                #     person_0_transform = np.dot(person_0_transform, self.offset_transform)
                t = translation_from_matrix(person_0_transform)
                q = quaternion_from_matrix(person_0_transform)
                # pose_stamped = PoseStamped()
                # pose_stamped.pose.position.x = t[0]
                # pose_stamped.pose.position.y = t[1]
                # pose_stamped.pose.position.z = t[2]
                # pose_stamped.pose.orientation.x = q[0]
                # pose_stamped.pose.orientation.y = q[1]
                # pose_stamped.pose.orientation.z = q[2]
                # pose_stamped.pose.orientation.w = q[3]
                # pose_stamped.header.frame_id = CAMERA_DEPTH_FRAME
                # pose_stamped.header.stamp = rospy.Time.now()
                # self.person_0_pub.publish(pose_stamped)
                self._tf_broadcaster.sendTransform(t, q, rospy.Time.now(), "person_1", REFERENCE_FRAME)

    def drawAxis(self, image, corners, image_points):
        corner = tuple(corners[0].ravel())

        image = cv2.line(image, corner, tuple(image_points[0].ravel()), (0, 0, 255), 3)
        image = cv2.line(image, corner, tuple(image_points[1].ravel()), (0, 255, 0), 3)
        image = cv2.line(image, corner, tuple(image_points[2].ravel()), (255, 0, 0), 3)

        return image

    def run(self, _):
        if self.camera_matrix is not None and self.camera_distortion is not None and self.image_np is not None:
            rospy.loginfo_throttle(1, "Going to find chessboard corners")
            # gray = cv2.cvtColor(self._image_np, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(self.image_np, chessboard_size, flags=cv2.CALIB_CB_FAST_CHECK)
            if ret:
                corners = np.flipud(corners)
                # rospy.loginfo_throttle(1, corners)
                # rospy.loginfo_throttle(1, corners.shape)
                # rospy.loginfo_throttle(1, self.chessboard_model.shape)
                # rospy.loginfo_throttle(1, "Chessboard corners found, going to solve pnp")

                dbg_img = self.image_np.copy()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.drawChessboardCorners(dbg_img, chessboard_size, corners, ret)
                # corners2 = cv2.cornerSubPix(self._image_np, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors.
                success, rvecs, tvecs, _ = cv2.solvePnPRansac(self.chessboard_model, corners, self.camera_matrix,
                                                              self.camera_distortion, flags=cv2.SOLVEPNP_ITERATIVE,
                                                              useExtrinsicGuess=False)
                # rospy.loginfo_throttle(1, "Rvec : {}; Tvec : {}".format(rvecs, tvecs))
                axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]).reshape(-1, 3)
                imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, self.camera_matrix, self.camera_distortion)
                dbg_img = self.drawAxis(dbg_img, corners, imgpts)
                self.debug_im_pub.publish(self.bridge.cv2_to_imgmsg(dbg_img, "mono8"))
                rot_q = self.rodriguez_to_quaternion(rvecs)
                self.image_chessboard_transform = transformation_matrix([tvecs[0], tvecs[1], tvecs[2]], [rot_q[0], rot_q[1], rot_q[2], rot_q[3]])
                self._tf_broadcaster.sendTransform(tvecs, rot_q, rospy.Time.now(), CHESSBOARD_IMAGE_FRAME, CAMERA_DEPTH_FRAME)
                self.calibrate()
                # msg = PoseStamped()s
                # msg.header.stamp = rospy.Time.now()
                # msg.header.frame_id = CAMERA_DEPTH_FRAME
                # msg.pose.position.x = tvecs[0]
                # msg.pose.position.y = tvecs[1]
                # msg.pose.position.z = tvecs[2]
                # msg.pose.orientation.x = rot_q[0]
                # msg.pose.orientation.y = rot_q[1]
                # msg.pose.orientation.z = rot_q[2]
                # msg.pose.orientation.w = rot_q[3]
                # self.chessboard_pub.publish(msg)

    def calibrate(self):
        if self.head_ref_transform is None or self.mocap_chessboard_transform is None or self.image_chessboard_transform is None:
            return
        self.offset_transform = concatenate_matrices(self.image_chessboard_transform, np.linalg.inv(self.mocap_chessboard_transform))
        #rospy.loginfo(self.offset_transform)


    # def callback_update_person_1(self, msg):
    #     if self.head_ref_transform is not None:
    #         person_0_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z],
    #                                                    [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz,
    #                                                     msg.pos[0].qw])
    #         person_0_transform = np.dot(np.linalg.inv(self.head_ref_transform), person_0_transform)
    #         t = translation_from_matrix(person_0_transform)
    #         q = quaternion_from_matrix(person_0_transform)
    #         pose_stamped = PoseStamped()
    #         pose_stamped.pose.position.x = t[0]
    #         pose_stamped.pose.position.y = t[1]
    #         pose_stamped.pose.position.z = t[2]
    #         pose_stamped.pose.orientation.x = q[0]
    #         pose_stamped.pose.orientation.y = q[1]
    #         pose_stamped.pose.orientation.z = q[2]
    #         pose_stamped.pose.orientation.w = q[3]
    #         pose_stamped.header.frame_id = "CameraDepth_optical_frame"
    #         pose_stamped.header.stamp = rospy.Time.now()
    #         self.pub_person_0.publish(pose_stamped)
    #
    # def callback_update_person_1(self, msg):
    #     if self.head_ref_transform is not None:
    #         person_0_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z],
    #                                                    [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz,
    #                                                     msg.pos[0].qw])
    #         person_0_transform = np.dot(np.linalg.inv(self.head_ref_transform), person_0_transform)
    #         t = translation_from_matrix(person_0_transform)
    #         q = quaternion_from_matrix(person_0_transform)
    #         pose_stamped = PoseStamped()
    #         pose_stamped.pose.position.x = t[0]
    #         pose_stamped.pose.position.y = t[1]
    #         pose_stamped.pose.position.z = t[2]
    #         pose_stamped.pose.orientation.x = q[0]
    #         pose_stamped.pose.orientation.y = q[1]
    #         pose_stamped.pose.orientation.z = q[2]
    #         pose_stamped.pose.orientation.w = q[3]
    #         pose_stamped.header.frame_id = "CameraDepth_optical_frame"
    #         pose_stamped.header.stamp = rospy.Time.now()
    #         self.pub_person_0.publish(pose_stamped)
    #
    # def callback_update_kinect_0(self, msg):
    #     if self.head_ref_transform is not None:
    #         person_0_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z], [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz, msg.pos[0].qw])
    #         person_0_transform = np.dot(np.linalg.inv(self.head_ref_transform), person_0_transform)
    #         t = translation_from_matrix(person_0_transform)
    #         q = quaternion_from_matrix(person_0_transform)
    #         pose_stamped = PoseStamped()
    #         pose_stamped.pose.position.x = t[0]
    #         pose_stamped.pose.position.y = t[1]
    #         pose_stamped.pose.position.z = t[2]
    #         pose_stamped.pose.orientation.x = q[0]
    #         pose_stamped.pose.orientation.y = q[1]
    #         pose_stamped.pose.orientation.z = q[2]
    #         pose_stamped.pose.orientation.w = q[3]
    #         pose_stamped.header.frame_id = "CameraDepth_optical_frame"
    #         pose_stamped.header.stamp = rospy.Time.now()
    #         self.pub_person_0.publish(pose_stamped)

    def callback_update_head_ref(self, msg):
        if len(msg.pos)>0:
            self.head_ref_transform = transformation_matrix([msg.pos[0].x, msg.pos[0].y, msg.pos[0].z], [msg.pos[0].qx, msg.pos[0].qy, msg.pos[0].qz, msg.pos[0].qw])
            if self.offset_transform is not None:
                t = translation_from_matrix(self.offset_transform)
                q = quaternion_from_matrix(self.offset_transform)
                self._tf_broadcaster.sendTransform(t, q, rospy.Time.now(), REFERENCE_FRAME, CAMERA_DEPTH_FRAME)



if __name__ == '__main__':
    rospy.init_node("mummer_mocap_recording_node")
    mmr = MummerMocapRecording()
    rospy.Timer(rospy.Duration(1), mmr.run)
    rospy.spin()

