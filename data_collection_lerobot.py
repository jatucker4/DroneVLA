#!/usr/bin/env python3
"""
Data collection node that subscribes to compressed image topics, decodes and resizes them to (256,256,3),
synchronizes state information (pose and gripper) along with third‐person views, and finally saves the
collected trajectories into a LeRobot dataset.
"""

import threading
import numpy as np
import pygame
import os
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Imports for saving as a LeRobot dataset.
import shutil
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from datetime import datetime

# Define desired image size
TARGET_SIZE = (256, 256)

class DataSyncDroneNode(Node):
    def __init__(self):
        super().__init__('data_sync_drone_node')
        self.bridge = CvBridge()

        # QoS settings
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        qos_profile_reliable = QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribers for state information using message_filters.
        self.pos_sub = Subscriber(self, PoseStamped, '/vrpn_mocap/fmu/pose', qos_profile=qos_profile)
        # self.gripper_sub = Subscriber(self, Float32, '/gripper_state', qos_profile=qos_profile)
        # Subscribers for third-person cameras (compressed images)
        self.third_cam1_sub = Subscriber(self, CompressedImage, '/camera1/image_compressed') #, qos_profile=qos_profile_reliable)
        self.third_cam2_sub = Subscriber(self, CompressedImage, '/camera2/image_compressed') #, qos_profile=qos_profile_reliable)
        self.third_cam3_sub = Subscriber(self, CompressedImage, '/camera3/image_compressed') #, qos_profile=qos_profile_reliable)

        # Synchronize state and third-person cameras (5 topics)
        self.sync = ApproximateTimeSynchronizer(
            [
             self.pos_sub, 
             self.third_cam1_sub, 
             self.third_cam2_sub, 
             self.third_cam3_sub
             ],
            queue_size=10,
            slop=0.1,
            allow_headerless=False
        )
        self.sync.registerCallback(self.synced_callback)
        self.traj_buffer = []
        # Drone cameras (front and wrist) are handled separately.
        # They are assumed to publish compressed images.
        self.create_subscription(
            Image,
            '/hires_front_small_color',
            self.drone_front_callback,
            1
        )
        self.create_subscription(
            Image,
            '/hires_down_small_color',
            self.drone_wrist_callback,
            1
        )
        self.create_subscription(Float32, "/gripper_state", self.gripper_callback, 1)

        # Buffers for drone images
        self.front_image_buffer = None
        self.wrist_image_buffer = None

        # Data storage for trajectory frames.
        # Each frame is a dict with keys: "front_image", "wrist_image", "3pov_1", "3pov_2", "3pov_3", "state", "actions"
        self.frames = []
        self.prev_state = None
        self.gripper_val = 0.0

        # Pygame initialization for keyboard events
        pygame.init()
        self.screen = pygame.display.set_mode((300, 200))
        self.is_recording = False

        self.get_logger().info("DataSyncDroneNode initialized and waiting for data...")

    def gripper_callback(self, msg):
        self.gripper_val = msg.data
        
    def drone_front_callback(self, msg: Image):
        try:
            # Decode compressed front image
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize to TARGET_SIZE
            self.front_image_buffer = cv2.resize(img, TARGET_SIZE)
        except Exception as e:
            self.get_logger().error(f"Drone front image decode failed: {e}")

    def drone_wrist_callback(self, msg: Image):
        try:
            # Decode compressed wrist image
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize to TARGET_SIZE
            self.wrist_image_buffer = cv2.resize(img, TARGET_SIZE)
        except Exception as e:
            self.get_logger().error(f"Drone wrist image decode failed: {e}")

    def synced_callback(self, 
                        pos_msg: PoseStamped,
                        third_cam1_msg: CompressedImage, 
                        third_cam2_msg: CompressedImage, 
                        third_cam3_msg: CompressedImage):
        # Only record if demo is running.
        # print("NSYNC")
        if not self.is_recording:
            return

        # Decode third-person camera images.
        try:
            third_person_1 = self.bridge.compressed_imgmsg_to_cv2(third_cam1_msg, desired_encoding='bgr8')
            third_person_2 = self.bridge.compressed_imgmsg_to_cv2(third_cam2_msg, desired_encoding='bgr8')
            third_person_3 = self.bridge.compressed_imgmsg_to_cv2(third_cam3_msg, desired_encoding='bgr8')
            # Resize to TARGET_SIZE
            third_person_1 = cv2.resize(third_person_1, TARGET_SIZE)
            third_person_2 = cv2.resize(third_person_2, TARGET_SIZE)
            third_person_3 = cv2.resize(third_person_3, TARGET_SIZE)
        except Exception as e:
            self.get_logger().error(f"Third-person image conversion failed: {e}")
            return

        # Ensure drone camera buffers are available.
        if self.front_image_buffer is None or self.wrist_image_buffer is None:
            self.get_logger().warn("Drone camera buffers not ready; skipping frame.")
            return

        # Compute state from pos_msg.
        pos = pos_msg.pose.position
        orient = pos_msg.pose.orientation
        # Extract position.
        x_pos, y_pos, z_pos = pos.x, pos.y, pos.z
        # Convert orientation quaternion to Euler angles.
        quat = np.array([orient.x, orient.y, orient.z, orient.w])
        try:
            # Using scipy to convert quaternion to Euler (roll, pitch, yaw)
            from scipy.spatial.transform import Rotation as R
            roll, pitch, yaw = R.from_quat(quat).as_euler('xyz', degrees=False)
        except Exception as e:
            self.get_logger().error(f"Quaternion conversion failed: {e}")
            return

        # Use gripper value from gripper_msg.
        gripper = 1.0 if self.gripper_val != 0.0 else 0.0

        # Construct an 7-element state vector.
        state = np.array([x_pos, y_pos, z_pos, yaw, gripper, 0.0, 0.0], dtype=np.float32)

        # Compute action as delta of the first 7 elements.
        if self.prev_state is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action = self.compute_action_delta(self.prev_state, state)
        self.prev_state = state.copy()

        # Construct frame data.
        frame = {
            "front_image": self.front_image_buffer,    # Drone front image (256x256x3)
            "wrist_image": self.wrist_image_buffer,      # Drone wrist image (256x256x3)
            "3pov_1": third_person_1,                    # Third-person camera 1 (256x256x3)
            "3pov_2": third_person_2,                    # Third-person camera 2 (256x256x3)
            "3pov_3": third_person_3,                    # Third-person camera 3 (256x256x3)
            "state": state,                              # 8-element state vector
            "actions": action,                            # 7-element action delta
            "task": "language_instruction"
        }
        self.frames.append(frame)
        self.get_logger().debug(f"Frame recorded, total frames: {len(self.frames)}")


    def angular_difference(self,yaw_current, yaw_prev):
        """
        Compute the minimal difference between two yaw angles (in radians)
        wrapping the difference to the range [-pi, pi].
        """
        delta = yaw_current - yaw_prev
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        return delta

    def compute_action_delta(self,state, prev_state):
        """
        Compute the delta (difference) between current and previous state vectors,
        taking into account the circular nature of yaw (assumed at index 3).

        Parameters:
            state (array-like): Current state vector.
            prev_state (array-like): Previous state vector.
            
        Returns:
            delta (np.ndarray): The computed action delta vector for the first 7 elements.
        """
        # Ensure the states are numpy arrays.
        state = np.array(state, dtype=np.float32)
        prev_state = np.array(prev_state, dtype=np.float32)

        # Compute the straightforward differences for the first 7 elements.
        delta = state[:7] - prev_state[:7]

        # Correct the yaw difference (index 3) using angular wrapping.
        delta[3] = self.angular_difference(state[3], prev_state[3])
        return delta


    def begin_demo(self):
        self.get_logger().info("Starting demonstration recording.")
        self.is_recording = True
        self.frames = []
        self.prev_state = None

    def end_demo(self):
        self.get_logger().info("Ending demonstration recording.")
        self.is_recording = False
        self.traj_buffer.append(self.frames)

    def save_demo(self):
        self.get_logger().info("Saving demonstration into LeRobot dataset...")

        # Define repository name for the dataset (modify as needed).
        REPO_NAME = "jatucker/dronevla_3_10_2025_take_off_hat"
        output_path = HF_LEROBOT_HOME / REPO_NAME
        if output_path.exists():
            shutil.rmtree(output_path)

        # Create LeRobot dataset with appropriate features.
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "3pov_1": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "3pov_2": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "3pov_3": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        # For each recorded frame, add a frame to the dataset.
        epi_index = 0
        for trajectory in self.traj_buffer:
            epi_length = len(trajectory)
            for frame in trajectory:
                dataset.add_frame({
                    "image": frame["front_image"],
                    "wrist_image": frame["wrist_image"],
                    "3pov_1": frame["3pov_1"],
                    "3pov_2": frame["3pov_2"],
                    "3pov_3": frame["3pov_3"],
                    "state": frame["state"],
                    "actions": frame["actions"],
                    "task" : "grab the chips and put them in the blue bucket"
                })
            dataset.save_episode()
            epi_index+=1
        # Consolidate the dataset.
        # dataset.consolidate(run_compute_stats=False)
        self.get_logger().info(f"Saved demonstration data to LeRobot dataset at {output_path}")

def monitor_keys(data_sync_node: DataSyncDroneNode):
    """
    Monitor keyboard events using pygame:
      - Press 'b' to begin recording.
      - Press 'e' to end recording.
      - Press 's' to save the recorded demo.
    """
    print("Press 'b' to begin, 'e' to end, and 's' to save the demonstration.")
    while rclpy.ok():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    data_sync_node.begin_demo()
                elif event.key == pygame.K_e:
                    data_sync_node.end_demo()
                elif event.key == pygame.K_s:
                    data_sync_node.save_demo()
        time.sleep(0.1)

def main():
    rclpy.init()
    data_sync_node = DataSyncDroneNode()

    # Start ROS spinning in a separate thread.
    def spin_ros():
        rclpy.spin(data_sync_node)
    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()

    # Start keyboard monitoring in another thread.
    key_thread = threading.Thread(target=monitor_keys, args=(data_sync_node,), daemon=True)
    key_thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        data_sync_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()
        key_thread.join()

if __name__ == '__main__':
    main()
