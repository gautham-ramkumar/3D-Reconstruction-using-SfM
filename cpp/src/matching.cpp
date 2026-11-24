#include "sfm/matching.hpp"

#include <algorithm>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "sfm/camera.hpp"
#include "sfm/rng.hpp"

namespace sfm {

std::pair<Points2D, Points2D> match_points(const std::vector<cv::DMatch>& matches,
                                           const std::vector<cv::KeyPoint>& kps_i,
                                           const std::vector<cv::KeyPoint>& kps_j) {
    Points2D p1, p2;
    p1.reserve(matches.size());
    p2.reserve(matches.size());
    for (const auto& m : matches) {
        p1.emplace_back(kps_i[m.queryIdx].pt.x, kps_i[m.queryIdx].pt.y);
        p2.emplace_back(kps_j[m.trainIdx].pt.x, kps_j[m.trainIdx].pt.y);
    }
    return {p1, p2};
}

FeatureSet extract_features_and_matches(const std::vector<cv::Mat>& images,
                                        const IntrinsicsMap& intrinsics,
                                        const MatchingParams& params) {
    set_random_seed(params.seed);

    auto sift = cv::SIFT::create(params.nfeatures, 3, params.contrast_threshold,
                                 params.edge_threshold);
    cv::BFMatcher matcher(cv::NORM_L2);

    FeatureSet out;
    for (const auto& img : images) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        sift->detectAndCompute(gray, cv::noArray(), kps, desc);
        out.keypoints.push_back(std::move(kps));
        out.descriptors.push_back(desc);
    }
    // Everything downstream (F, PnP, triangulation, BA) assumes a pinhole camera.
    if (params.undistort) undistort_keypoints(out.keypoints, intrinsics);

    const int num_images = static_cast<int>(images.size());
    const int window = params.window_size > 0 ? params.window_size : num_images;

    for (int i = 0; i < num_images; ++i) {
        for (int j = i + 1; j < std::min(i + 1 + window, num_images); ++j) {
            if (out.descriptors[i].empty() || out.descriptors[j].empty()) continue;

            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(out.descriptors[i], out.descriptors[j], knn, 2);

            std::vector<cv::DMatch> good;
            for (const auto& pair : knn) {
                if (pair.size() < 2) continue;  // knnMatch can return a single neighbour
                if (pair[0].distance < params.ratio_thresh * pair[1].distance) {
                    good.push_back(pair[0]);
                }
            }
            if (static_cast<int>(good.size()) < params.min_matches) continue;

            const auto [p1, p2] = match_points(good, out.keypoints[i], out.keypoints[j]);
            FundamentalResult res = ransac_fundamental_matrix(p1, p2, params.ransac_iterations,
                                                              params.ransac_threshold);
            if (params.refine_f) {
                res = refine_fundamental_matrix(res, p1, p2, params.ransac_threshold);
            }

            if (res.F && static_cast<int>(res.inliers.size()) > params.min_inliers) {
                std::vector<cv::DMatch> verified;
                verified.reserve(res.inliers.size());
                for (int k : res.inliers) verified.push_back(good[k]);
                out.matches[{i, j}] = std::move(verified);
                std::cout << "Verified pair (" << i << ", " << j << ") with "
                          << res.inliers.size() << " inliers.\n";
            }
        }
    }
    std::cout << out.matches.size() << " matches found.\n";
    return out;
}

}  // namespace sfm
