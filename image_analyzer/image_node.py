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

        # Subscribing to the depth image topic
        self.depth_sub = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )

        self.image_counter = 0  # Counter to keep track of how many images we've processed
        self.results = []       # Store angle and area results for CSV export

        # Set camera intrinsics manually (assuming standard pinhole model)
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(
            width=640, height=480,
            fx=525.0, fy=525.0,
            cx=319.5, cy=239.5
        )

        self.get_logger().info("Depth Analyzer Node Initialized.")

    def depth_callback(self, depth_msg):
        try:
            # Convert ROS depth message to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # Convert to Open3D image format
            depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

            # Creating a dummy RGB image (since we're only using depth)
            rgb_dummy = o3d.geometry.Image(np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8))

            # Create RGBD image (Open3D expects both even for point cloud generation)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_dummy, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1000.0, depth_trunc=3.0
            )

            # Generate point cloud from RGBD and intrinsics
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.intrinsics
            )

            # Downsample for performance and noise reduction
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd.remove_non_finite_points()
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Try to segment the largest plane (likely to be the face of the box)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
            plane_cloud = pcd.select_by_index(inliers)

            # Sanity check for enough inliers
            if len(inliers) < 50:
                self.get_logger().warn("Not enough plane inliers found.")
                return

            # Estimate normal angle between plane and camera z-axis
            normal_vec = np.array(plane_model[:3])
            angle_rad = np.arccos(np.clip(np.dot(normal_vec, [0, 0, 1]) / np.linalg.norm(normal_vec), -1.0, 1.0))
            normal_angle_deg = math.degrees(angle_rad)

            # Estimate surface area using convex hull of the inlier points
            try:
                hull, _ = plane_cloud.compute_convex_hull()
                area = hull.get_surface_area()
            except Exception as e:
                self.get_logger().error(f"Convex hull computation failed: {e}")
                area = 0.0  # If area can't be computed, set it to zero

            # Store result
            self.image_counter += 1
            self.results.append({
                'Image Number': self.image_counter,
                'Normal Angle (deg)': round(normal_angle_deg, 2),
                'Visible Area (m^2)': round(area, 4)
            })

            # Log result
            self.get_logger().info(
                f"Image {self.image_counter}: Normal Angle = {round(normal_angle_deg, 2)}°, Visible Area = {round(area, 4)} m²"
            )

            # Save results to CSV
            self.save_csv()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_csv(self):
        # Write the results to a CSV file (overwrite each time for safety)
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
