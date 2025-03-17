#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import sys
import torch
import pathlib
import dill
import hydra
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32, Float32MultiArray
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class DronePolicyNode(Node):
    def __init__(self):
        super().__init__('drone_policy_node')
        self.get_logger().info("Initializing DronePolicyNode.")
        # QoS settings
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        # --------------------------------------
        # Parameters
        # --------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.horizon = 1  # How many timesteps we buffer before we feed to the policy
        self.rate_hz = 10
        self.bridge = CvBridge()

        # Buffers
        self.pose_buffer = []
        self.gripper_buffer = []  # If your old policy expects a "gripper scalar"
        self.front_image_buffer = []
        self.down_image_buffer = []
        self.third_pov_2_buffer = []
        # You can add 3rd person cameras if needed.

        # --------------------------------------
        # Subscribers
        # --------------------------------------
        # 1) Drone pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/fmu/pose',
            self.pose_callback,
            qos_profile= qos_profile
        )

        # 2) Gripper "state" (optional if your policy needs a scalar)
        self.gripper_sub = self.create_subscription(
            Float32,
            '/gripper_state',
            self.gripper_callback,
            10
        )

        # 3) Drone camera(s) – adjust to your use-case
        # Example: front camera
        self.front_cam_sub = self.create_subscription(
            Image,  # or CompressedImage if you prefer
            '/hires_front_small_color',
            self.front_cam_callback,
            10
        )
        # Example: downward camera
        self.down_cam_sub = self.create_subscription(
            Image,
            '/hires_down_small_color',
            self.down_cam_callback,
            10
        )

        # Example: downward camera
        self.thirdpov2_cam_sub = self.create_subscription(
            CompressedImage,
            '/camera2/image_compressed',
            self.thirdpov2_callback,
            10
        )
        # --------------------------------------
        # Publisher
        # --------------------------------------
        # Publish velocity commands as Twist
        self.cmd_xyz_pub = self.create_publisher(
            Float32MultiArray,
            '/control',
            10
        )

        self.cmd_gripper = self.create_publisher(
            Float32,
            '/control_gripper',
            10
        )
        # --------------------------------------
        # Load policy
        # --------------------------------------
        self.client = websocket_client_policy.WebsocketClientPolicy(host="moraband.stanford.edu", port=8000)
        self.get_logger().info(f"Loaded pi Zero!")

        # Start a timer to run at self.rate_hz
        self.timer = self.create_timer(1.0 / self.rate_hz, self.run_policy)

    # -----------------------------
    # Subscribers' Callbacks
    # -----------------------------
    def pose_callback(self, msg: PoseStamped):
        """Callback to buffer the pose up to horizon length."""
        # For example, convert from quaternion to [x,y,z,roll,pitch,yaw] or
        # just store the raw [x,y,z,qx,qy,qz,qw].
        # Adjust to whatever your policy expects.
        # Compute state from pos_msg.
        pos = msg.pose.position
        orient = msg.pose.orientation
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

        pose_vec = np.array([x_pos, y_pos, z_pos, yaw])
        self.pose_buffer.append(pose_vec)
        if len(self.pose_buffer) > self.horizon:
            self.pose_buffer.pop(0)

    def gripper_callback(self, msg: Float32):
        # If your old policy code expects a float for the "gripper"
        # In a drone scenario, you might remove this altogether. 
        val = msg.data
        
        self.gripper_buffer.append(val)
        if len(self.gripper_buffer) > self.horizon:
            self.gripper_buffer.pop(0)

    def front_cam_callback(self, msg: Image):
        # If your policy needs images, convert them to torch Tensors or store them as numpy arrays. 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Example: we just store in a buffer – do any resizing or normalization for your policy
            self.front_image_buffer.append(cv_image)
            if len(self.front_image_buffer) > self.horizon:
                self.front_image_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Front cam callback error: {e}")

    def down_cam_callback(self, msg: Image):
        # Similarly for the downward camera
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.down_image_buffer.append(cv_image)
            if len(self.down_image_buffer) > self.horizon:
                self.down_image_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Down cam callback error: {e}")

    def thirdpov2_callback(self, msg: CompressedImage):
        # Similarly for the downward camera
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.resize(cv_image, (255,255))
            self.third_pov_2_buffer.append(cv_image)
            if len(self.third_pov_2_buffer) > self.horizon:
                self.third_pov_2_buffer.pop(0)
        except Exception as e:
            self.get_logger().error(f"Down cam callback error: {e}")

    # -----------------------------
    # Policy Step
    # -----------------------------
    def run_policy(self):
        """
        Called on a fixed timer (e.g. 20Hz). If we have enough samples 
        in the buffers, run the policy and publish an action as Twist.
        """
        # Check if we have enough data
        if len(self.pose_buffer) < self.horizon or len(self.gripper_buffer) < self.horizon or len(self.front_image_buffer) < self.horizon or len(self.third_pov_2_buffer) < self.horizon or len(self.down_image_buffer) < self.horizon:
            return
        # If images or gripper are needed, also check those buffers:
        # if len(self.front_image_buffer) < self.horizon: return
        # if len(self.gripper_buffer) < self.horizon: return

        # Build your observation dictionary the same shape as your training
        # For example:
        #   obs_dict["pose"] = (batch_size=1, horizon, 7) 
        #   obs_dict["front_img"] = (batch_size=1, horizon, H,W,3)
        #   ...
        # Here is a minimal example if your policy just uses pose:
        pose_np = np.stack(self.pose_buffer, axis=0)  # shape [horizon, 7]
        
        state_np = np.zeros((7,)).astype(np.float32)
        state_np[:4] = self.pose_buffer[-1]
        state_np[4] = self.gripper_buffer[-1]
        # shape [1, horizon, 7]

        obs_dict = {
            "observation/state": state_np,
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(self.front_image_buffer[-1],255,255)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(self.down_image_buffer[-1],255,255)),
            "observation/3pov_2": self.third_pov_2_buffer[-1],
            "prompt": "grab the chips and place them in the blue bin"
        }

        # If your policy also needs images, process them. Something like:
        # front_imgs_np = np.stack([some_preproc(img) for img in self.front_image_buffer], axis=0)
        # front_imgs_torch = torch.tensor(front_imgs_np, dtype=..., device=...).unsqueeze(0)
        # obs_dict["images"] = front_imgs_torch

        drone_action = self.client.infer(obs_dict)["actions"][-1,:]

        # For demonstration, let's say the first 4 elements are [vx, vy, vz, yaw_rate].
        del_x, del_y, del_z, yaw = drone_action[0], drone_action[1], drone_action[2], drone_action[3]

        cmd_msg = Float32MultiArray()
        cmd_msg.data = [state_np[0] + del_x, state_np[1] + del_y, state_np[2] + del_z]
        
        self.cmd_xyz_pub.publish(cmd_msg)

        gripper_val = -1 if drone_action[-1] < 0 else 1
        cmd_gripper_msg = Float32()
        cmd_gripper_msg.data = float(gripper_val)
        self.cmd_gripper.publish(cmd_gripper_msg)

        self.get_logger().info(
            f"Published Drone action: x={state_np[0] + del_x:.3f}, y={state_np[1] + del_y:.3f}, z={state_np[2] + del_z:.3f}, yaw={yaw:.3f}, gripper={gripper_val}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = DronePolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
