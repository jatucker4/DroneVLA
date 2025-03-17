#!/usr/bin/env python3
"""
A ROS2 node that captures images from multiple RealSense cameras and publishes them as sensor_msgs/CompressedImage.
Each camera will be named camera1, camera2, ..., cameraN.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')

        self.bridge = CvBridge()

        # List all connected RealSense devices
        self.pipeline_map = {}  # Dictionary to store pipelines for each camera
        self.publisher_map = {}  # Dictionary to store publishers for each camera

        # Initialize device manager and get all connected devices
        self.device_list = self.get_connected_devices()
        if not self.device_list:
            self.get_logger().error("No RealSense devices connected.")
            return

        self.get_logger().info(f"Found {len(self.device_list)} devices.")

        # Create a pipeline and publisher for each connected camera
        self.camera_counter = 1  # Start numbering cameras from 1
        for device in self.device_list:
            self.setup_device(device)

        # Set a timer to periodically capture and publish frames for each camera
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def get_connected_devices(self):
        """Returns a list of connected RealSense devices."""
        ctx = rs.context()
        devices = ctx.query_devices()
        device_list = []
        for device in devices:
            device_list.append(device)
            serial_number = device.get_info(rs.camera_info.serial_number)
            self.get_logger().info(f"Found device with serial number: {serial_number}")
        return device_list

    def setup_device(self, device):
        """Set up the pipeline and publisher for each connected device."""
        serial_number = device.get_info(rs.camera_info.serial_number)

        # Create a unique topic name based on the camera number (camera1, camera2, ...)
        topic_name = f"/camera{self.camera_counter}/image_raw"
        self.publisher_map[self.camera_counter] = self.create_publisher(CompressedImage, topic_name, 1)
        self.get_logger().info(f"Publishing on topic: {topic_name}")
        
        # Set up the pipeline for the camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.mjpeg, 30)
        pipeline.start(config)
        self.pipeline_map[self.camera_counter] = pipeline
        self.get_logger().info(f"Started pipeline for camera {self.camera_counter}.")

        # Increment the camera counter for the next camera
        self.camera_counter += 1

    def timer_callback(self):
        """Callback function that is called periodically to capture and publish images."""
        for camera_id, pipeline in self.pipeline_map.items():
            try:
                # Wait for the next set of frames from the camera
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                self.get_logger().error(f"Error getting frames from camera {camera_id}: {e}")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                self.get_logger().warning(f"No color frame received from camera {camera_id}.")
                continue

            # Convert RealSense frame data to a numpy array
            compressed_data = np.asanyarray(color_frame.get_data()).tobytes()

            # Convert the image to a ROS2 CompressedImage message
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.header.frame_id = f"camera{camera_id}"
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed_data

            # Publish the message
            self.publisher_map[camera_id].publish(compressed_msg)
            self.get_logger().debug(f"Published image frame from camera {camera_id}.")

    def destroy_node(self):
        """Clean up and stop the pipelines."""
        for pipeline in self.pipeline_map.values():
            pipeline.stop()
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
