import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import pandas as pd
import os
import math
import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber

class DepthAnalyzerNode(Node):
    def __init__(self):
        super().__init__('depth_analyzer')
        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )

        self.image_counter = 0
        self.results = []

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(
            width=640, height=480,
            fx=525.0, fy=525.0,
            cx=319.5, cy=239.5
        )

        self.get_logger().info("Depth Analyzer Node Initialized.")

    def depth_callback(self, depth_msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

            rgb_dummy = o3d.geometry.Image(np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_dummy, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1000.0, depth_trunc=3.0
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.intrinsics
            )

            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd.remove_non_finite_points()
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Segment largest plane (likely box face)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
            plane_cloud = pcd.select_by_index(inliers)

            if len(inliers) < 50:
                self.get_logger().warn("Not enough plane inliers found.")
                return

            # Normal angle
            normal_vec = np.array(plane_model[:3])
            angle_rad = np.arccos(np.clip(np.dot(normal_vec, [0, 0, 1]) / np.linalg.norm(normal_vec), -1.0, 1.0))
            normal_angle_deg = math.degrees(angle_rad)

            # Estimate area via convex hull of plane
            try:
                hull, _ = plane_cloud.compute_convex_hull()
                area = hull.get_surface_area()
            except Exception as e:
                self.get_logger().error(f"Convex hull computation failed: {e}")
                area = 0.0

            self.image_counter += 1
            self.results.append({
                'Image Number': self.image_counter,
                'Normal Angle (deg)': round(normal_angle_deg, 2),
                'Visible Area (m^2)': round(area, 4)
            })

            self.get_logger().info(
                f"Image {self.image_counter}: Normal Angle = {round(normal_angle_deg, 2)}°, Visible Area = {round(area, 4)} m²"
            )

            self.save_csv()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv("depth_analysis_results.csv", index=False)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnalyzerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

