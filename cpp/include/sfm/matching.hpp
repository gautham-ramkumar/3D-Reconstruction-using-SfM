// SIFT feature extraction and geometrically verified pairwise matching.
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "sfm/geometry.hpp"
#include "sfm/types.hpp"

namespace sfm {

struct MatchingParams {
    int nfeatures = 8000;            // max SIFT features per image; 0 = unlimited
    int window_size = 8;             // match pairs with j - i <= window_size; <= 0 = all pairs
    double ratio_thresh = 0.8;       // Lowe ratio test
    int min_matches = 15;            // ratio-test matches needed before RANSAC
    int min_inliers = 15;            // pair kept only if inliers > min_inliers
    int ransac_iterations = 2000;
    double ransac_threshold = 1.0;   // Sampson error, px^2
    bool refine_f = true;            // refit F on inliers (not in the Python version)
    bool undistort = true;           // remove k1 distortion from keypoints before matching
    // Lower than OpenCV's 0.04 to pick up more features on low-texture surfaces (Buddha).
    double contrast_threshold = 0.02;
    double edge_threshold = 10.0;
    unsigned int seed = 0;
};

struct FeatureSet {
    Keypoints keypoints;  // undistorted pixel positions when params.undistort is set
    Descriptors descriptors;
    PairMatches matches;  // only geometrically verified inlier matches
};

FeatureSet extract_features_and_matches(const std::vector<cv::Mat>& images,
                                        const IntrinsicsMap& intrinsics,
                                        const MatchingParams& params = {});

// Pixel positions of a pair's matches: query keypoints in image i, train keypoints in j.
std::pair<Points2D, Points2D> match_points(const std::vector<cv::DMatch>& matches,
                                           const std::vector<cv::KeyPoint>& kps_i,
                                           const std::vector<cv::KeyPoint>& kps_j);

}  // namespace sfm
