# 3D RECONSTRUCTION USING SFM
A robust Incremental Structure from Motion (SfM) pipeline implemented in Python. This project reconstructs sparse 3D geometry from monocular image sequences by estimating camera poses and triangulating 3D points. It utilizes OpenCV for feature extraction and sequential PnP tracking, and GTSAM for global Bundle Adjustment to optimize the structure and minimize drift.

# Tech Stack
- Language: Python
- Computer Vision: OpenCV (cv2)
- Optimization: GTSAM (Georgia Tech Smoothing and Mapping library)
- Numerical Computing: NumPy
- Visualization: Open3D, Matplotlib

# OUTPUTS
| Pre-Optimization | Post-Optimization |
| :---: | :---: |
| <img src="Outputs/Output1.png" width="100%"> | <img src="Outputs/Output2.png" width="100%"> |

# Video Demo
<video 
  src="Outputs/SfM_Demo.webm"
  width="900"
  controls
  autoplay
  loop
  muted
  style="border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,0.25);">
</video>
