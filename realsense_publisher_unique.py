#!/usr/bin/env python3
"""
A ROS2 node that captures images from a RealSense camera and publishes them as sensor_msgs/Image.
Each instance uses a unique camera_id to differentiate its messages.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        # Declare a parameter for a unique camera ID; default is 'camera1'
        self.declare_parameter('camera_id', 'camera1')
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().string_value
        self.get_logger().info(f"Camera ID set to: {self.camera_id}")

        # self.camera_id = 10

        # Use the camera_id in the topic name (e.g., /camera1/image_raw)
        topic_name = f"/{self.camera_id}/image_raw"
        self.publisher_ = self.create_publisher(Image, topic_name, 1)
        self.get_logger().info(f"Publishing on topic: {topic_name}")

        self.bridge = CvBridge()

        # Set a timer to periodically capture and publish frames.
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Configure and start the RealSense pipeline.
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Enable the color stream at 640x480, 30 fps.
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.get_logger().info("RealSense pipeline started.")

    def timer_callback(self):
        try:
            # Wait for the next set of frames from the camera.
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as e:
            self.get_logger().error(f"Error getting frames: {e}")
            return

        color_frame = frames.get_color_frame()
        if not color_frame:
            self.get_logger().warning("No color frame received.")
            return

        # Convert RealSense frame data to a numpy array.
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the image to a ROS2 Image message.
        msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        # Use the camera_id in the frame_id
        msg.header.frame_id = self.camera_id

        # Publish the message.
        self.publisher_.publish(msg)
        self.get_logger().debug(f"Published image frame from {self.camera_id}.")

    def destroy_node(self):
        # Stop the RealSense pipeline when shutting down.
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

