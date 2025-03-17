import threading
import numpy as np
import pygame
import os
import tqdm
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool, Float32MultiArray
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class DataSyncDroneNode(Node):
    def __init__(self):
        super().__init__('data_sync_drone_node')
        self.bridge = CvBridge()

        # Define the QoS settings (Reliable)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Ensure reliable delivery
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        qos_profile2 = QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribers for drone state topics
        self.pos_sub = Subscriber(self, PoseStamped, '/vrpn_mocap/fmu/pose', qos_profile=qos_profile)
        # self.pos_sub_vehicle = self.create_subscription(PoseStamped, '/fmu/out/vehicle_odometry', self.pos_vehicle_callback, qos_profile)
        self.gripper_sub = self.create_subscription(Float32, '/gripper_state', self.gripper_callback, 1)#, qos_profile)

        # # Subscribers for drone cameras (we will use camera1 and camera2 for our dataset)
        # self.drone_cam1_sub = Subscriber(self, Image, '/front_cam', qos_profile=qos_profile)
        # self.drone_cam2_sub = Subscriber(self, Image, '/down_cam', qos_profile=qos_profile)

        self.drone_cam1_sub = self.create_subscription(Image, '/hires_front_small_color', self.drone_cam1_callback, 1)#, qos_profile)
        self.drone_cam2_sub = self.create_subscription(Image, '/hires_down_small_color', self.drone_cam2_callback, 1)#, qos_profile)

        # Subscribers for third-person cameras (not used for dataset storage here but synced)
        self.third_cam1_sub = Subscriber(self, Image, '/camera1/image_raw')#, qos_profile=qos_profile2)
        self.third_cam2_sub = Subscriber(self, Image, '/camera2/image_raw')#, qos_profile=qos_profile2)
        self.third_cam3_sub = Subscriber(self, Image, '/camera3/image_raw')#, qos_profile=qos_profile2)

        # We set up an ApproximateTimeSynchronizer for all eight topics
        self.sync = ApproximateTimeSynchronizer(
            [
            # self.pos_sub, 
            # self.pos_sub_vehicle, 
            # self.gripper_sub,
            # self.drone_cam1_sub, 
            # self.drone_cam2_sub,
            self.third_cam1_sub, self.third_cam2_sub, self.third_cam3_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=False
        )
        self.sync.registerCallback(self.synced_callback)

        # Data storage
        self.frames = []  # each element is a dict with keys: "image", "wrist_image", "state", "actions"
        self.prev_state = None

        # # Pygame initialization for keyboard events (to begin, end, and save demo)
        pygame.init()
        self.screen = pygame.display.set_mode((300, 200))
        self.is_recording = False

        self.get_logger().info("DataSyncDroneNode initialized and waiting for data...")

    def gripper_callback(self, msg):
        return
        self.gripper_val = msg.data

    def drone_cam1_callback(self, msg):
        return
        image_drone = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.front_image_buffer = np.zeros((255,255,3))

    def drone_cam2_callback(self, msg):
        return
        wrist_image_drone = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.wrist_image_buffer = np.zeros((255,255,3))

    def synced_callback(self, #pos_msg, 
                        # drone_pos_msg, 
                        # gripper_msg,
                        # drone_cam1_msg, 
                        # drone_cam2_msg, 
                        third_cam1_msg, third_cam2_msg, third_cam3_msg):
        print("NSYNCING", third_cam1_msg.header.stamp)
        # Only record if the demo is running
        if not self.is_recording:
            del third_cam1_msg, third_cam2_msg, third_cam3_msg
            return

        # Convert drone cameras to CV images
        try:
            third_person_1 = self.bridge.imgmsg_to_cv2(third_cam1_msg, desired_encoding='bgr8')
            third_person_2 = self.bridge.imgmsg_to_cv2(third_cam2_msg, desired_encoding='bgr8')
            third_person_3 = self.bridge.imgmsg_to_cv2(third_cam3_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error("Image conversion failed: {}".format(e))
            return

        # Optionally, resize images to 256x256 (as expected by LeRobot format)
        # image_drone = cv2.resize(image_drone, (256, 256))
        # wrist_image_drone = cv2.resize(wrist_image_drone, (256, 256))
        image_drone = self.front_image_buffer
        wrist_image_drone = self.wrist_image_buffer
        # Construct state vector.
        # We use drone position (x, y, z), yaw, gripper (as 1.0 for True, 0.0 for False).
        # To get an 8-element state, we pad with three zeros.
        pos = pos_msg.pose.position
        x, y, z = pos.x, pos.y, pos.z
        w, x, y, z = pos_msg.orientation.w, pos_msg.orientation.x, pos_msg.orientation.y, pos_msg.orientation.z
        quaternion = np.array([x,y,z,w])
        roll, pitch, yaw = R.from_quat(quaternion).as_euler('xyz', degrees=False)
        if self.gripper_val == -1 or self.gripper_val == 0.0:
            gripper = 0.0
        else:
            gripper = 1.0 

        state = np.array([x, y, z, yaw, gripper, 0.0, 0.0], dtype=np.float32)

        # Compute action as the delta between the current and previous state.
        # For action we take the first 7 elements (e.g. ignore the final padded zero).
        if self.prev_state is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action = (state[:7] - self.prev_state[:7]).astype(np.float32)
        self.prev_state = state.copy()

        # Store the frame data in our format
        frame = {
            "front_image": image_drone,         # from /drone/camera1
            "wrist_image": wrist_image_drone,  # from /drone/camera2
            "3pov_1": third_person_1,
            "3pov_2": third_person_2,
            "3pov_3":third_person_3,
            "state": state,               # 8-element state vector
            "actions": action             # 7-element delta
        }
        self.frames.append(frame)
        self.get_logger().debug("Frame recorded, total frames: {}".format(len(self.frames)))

    def begin_demo(self):
        self.get_logger().info("Starting demonstration recording.")
        self.is_recording = True
        self.frames = []
        self.prev_state = None

    def end_demo(self):
        self.get_logger().info("Ending demonstration recording.")
        self.is_recording = False

    def save_demo(self):
        # In this example we save the data as a numpy file.
        # In practice, you might convert this into the LeRobot format.
        save_path = os.path.join(os.getcwd(), "drone_demo_data.npz")
        # Convert lists into arrays or save as a dict of lists.
        # For simplicity, we assume each key is a list of frames.
        images = np.array([f["image"] for f in self.frames])
        wrist_images = np.array([f["wrist_image"] for f in self.frames])
        states = np.array([f["state"] for f in self.frames])
        actions = np.array([f["actions"] for f in self.frames])
        third_person_1 = np.array([f["3pov_1"] for f in self.frames])
        third_person_2 = np.array([f["3pov_2"] for f in self.frames])
        third_person_3 = np.array([f["3pov_3"] for f in self.frames])
        np.savez_compressed(save_path,
                            image=images,
                            third_person_1 = third_person_1,
                            third_person_2 = third_person_2,
                            third_person_3 = third_person_3,
                            wrist_image=wrist_images,
                            state=states,
                            actions=actions
        )   
        self.get_logger().info("Saved demonstration data to {}".format(save_path))


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

    # Start ROS spinning in a separate thread
    def spin_ros():
        rclpy.spin(data_sync_node)
    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()

    # Start keyboard monitoring in another thread
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
