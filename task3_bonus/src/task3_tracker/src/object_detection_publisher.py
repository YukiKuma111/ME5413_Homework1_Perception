#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import os
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image as PILImage
import numpy as np
from vision_msgs.msg import Detection2D, BoundingBox2D
import json

class ObjectDetectionTrackerROS:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('object_detection_tracker', anonymous=True)

        # Parameters
        self.base_input_directory = rospy.get_param('~base_input_directory', '../../../data')
        self.seq_num = rospy.get_param('~seq_num', 2)

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Define device (prefer GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # Load ground truth and first track bounding boxes
        self.gt_bboxes = self.load_bbox("groundtruth.txt")
        self.ft_bboxes = self.load_bbox("firsttrack.txt")[0]

        # Load DETR model and image processor
        try:
            rospy.loginfo("Loading DETR model and image processor...")
            self.image_processor = AutoImageProcessor.from_pretrained("./detr-resnet-50", use_fast=True)
            self.model = DetrForObjectDetection.from_pretrained("./detr-resnet-50", ignore_mismatched_sizes=True)
            self.model.to(self.device)
            rospy.loginfo("DETR model loaded and moved to device successfully.")
        except Exception as e:
            rospy.logerr(f"Error loading DETR model: {e}")
            exit(1)

        # Initialize previous bounding box
        self.prev_bbox = None

        # ROS Subscribers and Publishers
        self.image_sub = rospy.Subscriber('/me5413/image_raw', Image, self.image_callback)
        self.gt_pub = rospy.Publisher('/me5413/groundtruth', Detection2D, queue_size=10)
        self.track_pub = rospy.Publisher('/me5413/track', Detection2D, queue_size=10)

        # self.viz_pub = rospy.Publisher('/me5413/viz_output', Image, queue_size=50)

        # Publisher for NUSNET ID
        self.id_pub = rospy.Publisher('/me5413/nusnetID', String, queue_size=10)

        # Start publishing NUSNET ID at 5Hz
        self.publish_nusnet_id()
        
        rospy.loginfo("Object Detection Tracker Initialized")
        rospy.spin()

    def load_bbox(self, filename):
        """Load bounding boxes from the given file."""
        file = os.path.join(self.base_input_directory, f"seq{self.seq_num}", filename)
        if not os.path.exists(file):
            rospy.logwarn(f"{filename} file not found: {file}")
            return []

        with open(file, 'r') as f:
            bboxes = [list(map(int, line.strip().split(','))) for line in f.readlines()]
        rospy.loginfo(f"Loaded {filename}")

        return bboxes

    # def draw_bbox_on_image(self, image, bbox, r, g, b, label, thickness=2):
    #     """Draw bounding box and center point on the image."""
    #     x, y, w, h = map(int, bbox)
    #     center_x, center_y = x + w // 2, y + h // 2

    #     # Create a copy of the image to make it writable
    #     image = np.copy(image)

    #     # Draw the bounding box and label
    #     color = (r, g, b)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    #     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    #     cv2.circle(image, (center_x, center_y), radius=4, color=color, thickness=-1)

    #     return image
    
    def create_detection2d_msg(self, bbox, frame_id):
        """
        Convert bbox to Detection2D message.
        bbox: [x, y, width, height]
        """
        detection_msg = Detection2D()
        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.header.frame_id = str(frame_id)

        # Create bounding box
        bbox_msg = BoundingBox2D()
        bbox_msg.center.x = bbox[0] + bbox[2] / 2  # center_x
        bbox_msg.center.y = bbox[1] + bbox[3] / 2  # center_y
        bbox_msg.size_x = bbox[2]  # width
        bbox_msg.size_y = bbox[3]  # height

        detection_msg.bbox = bbox_msg

        return detection_msg


    def image_callback(self, msg):
        """Callback function to process incoming images."""
        try:
            # Convert ROS Image message to OpenCV image
            if msg.encoding == '8UC3':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Convert OpenCV image to PIL image
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        try:
            # Preprocess image and move to the appropriate device
            inputs = self.image_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
        except AttributeError as e:
            rospy.logerr(f"Model or processor not loaded: {e}")
            return
        except Exception as e:
            rospy.logerr(f"Error during model inference: {e}")
            return

        # Post-process detection results
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.75)[0]

        # Select the closest bounding box to the previous one
        selected_box = self.select_closest_bbox(results["boxes"].cpu())

        # Update previous bounding box or use fallback
        if selected_box is not None:
            min_x, min_y, max_x, max_y = selected_box.tolist()
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            self.prev_bbox = bbox
        elif self.prev_bbox is not None:
            bbox = self.prev_bbox
        else:
            bbox = [0, 0, 0, 0]  # No detection

        # Publish the tracking bounding box
        # self.track_pub.publish(json.dumps({"bbox": bbox}))
        track_msg = self.create_detection2d_msg(bbox, msg.header.seq)
        self.track_pub.publish(track_msg)

        # Visualize tracking bounding box
        # cv_image = self.draw_bbox_on_image(cv_image, bbox, 0, 255, 0, "Track")

        # Draw and publish ground truth bounding box if available
        frame_id = int(msg.header.seq)
        if frame_id < len(self.gt_bboxes):
            gt_bbox = self.gt_bboxes[frame_id]
            gt_msg = self.create_detection2d_msg(gt_bbox, msg.header.seq)
            self.gt_pub.publish(gt_msg)
            # cv_image = self.draw_bbox_on_image(cv_image, gt_bbox, 255, 0, 0, "GT")

        # # Publish visualization image
        # viz_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        # self.viz_pub.publish(viz_msg)

    def select_closest_bbox(self, boxes):
        """Select the bounding box closest to the previous one."""
        if self.prev_bbox is None:
            self.prev_bbox = self.ft_bboxes

        prev_x, prev_y, prev_w, prev_h = self.prev_bbox
        prev_center = (prev_x + prev_w / 2, prev_y + prev_h / 2)

        min_distance = float('inf')
        selected_box = None

        # Calculate Euclidean distance from previous bounding box
        for box in boxes:
            min_x, min_y, max_x, max_y = box.tolist()
            center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
            distance = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                selected_box = box

        # Return the closest box if within threshold, else None
        return selected_box if min_distance < 50 else None
    
    def publish_nusnet_id(self):
        """Publish NUSNET ID at 5Hz."""
        rate = rospy.Rate(5)  # 5Hz
        nusnet_id = String(data="E1373124")
        while not rospy.is_shutdown():
            self.id_pub.publish(nusnet_id)
            rate.sleep()

if __name__ == '__main__':
    try:
        ObjectDetectionTrackerROS()
    except rospy.ROSInterruptException:
        pass
