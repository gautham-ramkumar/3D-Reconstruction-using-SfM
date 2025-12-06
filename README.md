# 3D RECONSTRUCTION USING SFM
A robust Incremental Structure from Motion (SfM) pipeline implemented in Python. This project reconstructs sparse 3D geometry from monocular image sequences by estimating camera poses and triangulating 3D points. It utilizes OpenCV for feature extraction and sequential PnP tracking, and GTSAM for global Bundle Adjustment to optimize the structure and minimize drift.

# Tech Stack
- Language: Python
- Computer Vision: OpenCV (cv2)
- Optimization: GTSAM (Georgia Tech Smoothing and Mapping library)
- Numerical Computing: NumPy
- Visualization: Open3D, Matplotlib

# OUTPUTS
<img width="1740" height="1041" alt="Pre_optimzed_output1" src="https://github.com/user-attachments/assets/bb092119-7701-4233-9c15-42fcb8208ff2" />

<img width="1740" height="1041" alt="GTSAM_output1" src="https://github.com/user-attachments/assets/6ccfd0b6-7501-4a03-b80a-640d8012eb58" />
