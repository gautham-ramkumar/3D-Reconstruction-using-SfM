// Data structures shared by every stage of the SfM pipeline.
#pragma once

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace sfm {

struct Intrinsics {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    double k1 = 0.0;  // SIMPLE_RADIAL distortion; stored for BA / export
    int width = 0;
    int height = 0;
};

// World-to-camera transform: x_cam = R * X_world + t.
struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    Eigen::Vector3d center() const { return -R.transpose() * t; }
};

// Keyed by COLMAP camera ID (1-based). Image i uses camera ID i + 1.
using IntrinsicsMap = std::map<int, Intrinsics>;

using ImagePair = std::pair<int, int>;  // (i, j) with i < j
using Observation = std::pair<int, int>;  // (image index, keypoint index)

using Keypoints = std::vector<std::vector<cv::KeyPoint>>;   // per image
using Descriptors = std::vector<cv::Mat>;                   // per image
using PairMatches = std::map<ImagePair, std::vector<cv::DMatch>>;

using Tracks = std::map<int, std::vector<Observation>>;     // track_id -> observations
using KpToTrack = std::vector<std::unordered_map<int, int>>;  // per image: kp_idx -> track_id

// std::map (not unordered) so iteration order is deterministic across runs.
using Map3D = std::map<int, Eigen::Vector3d>;               // track_id -> 3D point
using CameraPoses = std::map<int, Pose>;                    // image index -> pose

}  // namespace sfm
