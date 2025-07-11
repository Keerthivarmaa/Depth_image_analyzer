# Depth_image_analyzer

This ROS 2 node analyzes depth images from a bag file to estimate:

- **Normal angle** 
- **Visible surface area** 

It publishes this data and saves the results in a CSV file.

---

## Approach

1. **Depth to Point Cloud**  
   Convert incoming `sensor_msgs/Image` depth image into a 3D point cloud using camera intrinsics.

2. **Preprocessing**  
   - Filter out invalid (`NaN`/zero) depth values.
   - Use `voxel_down_sample()` to reduce noise.
   

3. **Plane Segmentation**  
   - Apply RANSAC to detect the largest plane in the scene — assumed to be the most visible face of the cuboidal box.

4. **Normal Estimation**  
   - Estimate the surface normal of the detected plane.
   - Compute the angle between this normal and the camera's optical axis (Z-axis).

5. **Area Estimation**  
   - Create a mesh of the segmented plane.
   - Compute the surface area in square meters.

6. **Results Storage**  
   - For each depth frame, log:
     - Frame number
     - Estimated normal angle (degrees)
     - Visible surface area (m²)


---

## Requirements

- ROS 2 (Humble recommended)
- Python 3.8+
- Dependencies:
  - `open3d`
  - `numpy`
  - `cv_bridge`
  - `sensor_msgs`
  - `rosbag2_py`
  - `rclpy`


## How to Use

1. Clone and Build

         mkdir -p ~/ros2_ws/src

         cd ~/ros2_ws/src
         
         git clone <this_repo_url>
         
         cd ~/ros2_ws
         
         colcon build
         
         source install/setup.bash
	
2. Run the Node with ROS 2 Bag
 
         ros2 run image_analyzer image_node 
         
         ros2 bag play src/Depth_image_analyzer/depth
	
Ensure that your ROS 2 bag is being played.

## Output

Terminal prints:

	[INFO]  Image 3: Normal Angle = 47.35°, Visible Area = 0.142 m²
	
Saved CSV file format:

| Image Number | Normal Angle (deg) | Visible Area (m^2) |
| :----------- | :----------------- | :----------------- |
| 1            | 64.7               | 8.6452             |
| 2            | 14.49              | 2.0921             |
| 3            | 34.95              | 2.2839             |
| 4            | 125.8              | 3.2039             |
| 5            | 30.1               | 1.3237             |
| 6            | 49                 | 2.976              |
| 7            | 48.96              | 2.5923             |
   



