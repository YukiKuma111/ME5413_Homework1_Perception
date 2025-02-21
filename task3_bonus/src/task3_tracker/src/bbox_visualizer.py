#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from vision_msgs.msg import Detection2D

class BBoxVisualizer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('bbox_visualizer', anonymous=True)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize bounding box containers
        self.gt_bbox = None
        self.track_bbox = None

        # ROS Subscribers
        rospy.Subscriber('/me5413/groundtruth', Detection2D, self.gt_callback)
        rospy.Subscriber('/me5413/track', Detection2D, self.track_callback)
        rospy.Subscriber('/me5413/image_raw', Image, self.image_callback)

        # ROS Publisher for visualization
        self.viz_pub = rospy.Publisher('/me5413/viz_output', Image, queue_size=50)

        rospy.loginfo("Bounding Box Visualizer Initialized")
        rospy.spin()

    def gt_callback(self, msg):
        """Callback to update ground truth bounding box."""
        self.gt_bbox = [
            int(msg.bbox.center.x - msg.bbox.size_x / 2),
            int(msg.bbox.center.y - msg.bbox.size_y / 2),
            int(msg.bbox.size_x),
            int(msg.bbox.size_y)
        ]

    def track_callback(self, msg):
        """Callback to update tracking bounding box."""
        self.track_bbox = [
            int(msg.bbox.center.x - msg.bbox.size_x / 2),
            int(msg.bbox.center.y - msg.bbox.size_y / 2),
            int(msg.bbox.size_x),
            int(msg.bbox.size_y)
        ]

    def image_callback(self, msg):
        """Callback to process and visualize images."""
        try:
            if msg.encoding == '8UC3':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Draw ground truth bbox
        if self.gt_bbox:
            cv_image = self.draw_bbox(cv_image, self.gt_bbox, (255, 0, 0), 'GT')

        # Draw tracking bbox
        if self.track_bbox:
            cv_image = self.draw_bbox(cv_image, self.track_bbox, (0, 255, 0), 'Track')

        # Publish visualization
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error while publishing: {e}")

    def draw_bbox(self, image, bbox, color, label):
        """Draw bounding box with label on the image."""
        x, y, w, h = bbox
        center_x, center_y = x + w // 2, y + h // 2

        # Create a copy of the image to make it writable
        image = np.copy(image)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(image, (center_x, center_y), radius=4, color=color, thickness=-1)
        return image

if __name__ == '__main__':
    try:
        BBoxVisualizer()
    except rospy.ROSInterruptException:
        pass
