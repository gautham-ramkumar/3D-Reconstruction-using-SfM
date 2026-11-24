// Module 3 check: run SIFT + verified matching on the real dataset and sanity-check output.
// Usage: test_matching <data_dir> [--no-refine] [--no-undistort]
// Prints "PAIR i j inliers" lines so the counts can be diffed against the Python run.
#include <chrono>
#include <iostream>
#include <set>
#include <string>

#include "sfm/io.hpp"
#include "sfm/matching.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : ".";
    sfm::MatchingParams params;
    for (int a = 2; a < argc; ++a) {
        if (std::string(argv[a]) == "--no-refine") params.refine_f = false;
        if (std::string(argv[a]) == "--no-undistort") params.undistort = false;
    }
    std::cout << std::boolalpha << "refine_f = " << params.refine_f
              << ", undistort = " << params.undistort << "\n";

    const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
    const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);

    const auto t0 = std::chrono::steady_clock::now();
    const sfm::FeatureSet fs = sfm::extract_features_and_matches(set.images, intr, params);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    size_t total_kps = 0;
    for (const auto& k : fs.keypoints) total_kps += k.size();
    std::cout << "Keypoints/image (avg): " << total_kps / fs.keypoints.size() << "\n";
    std::cout << "Matching took " << secs << " s\n";

    bool pairs_ok = true, indices_ok = true, counts_ok = true, unique_ok = true;
    for (const auto& [pair, ms] : fs.matches) {
        const auto [i, j] = pair;
        std::cout << "PAIR " << i << " " << j << " " << ms.size() << "\n";
        pairs_ok &= i < j && j - i <= params.window_size;
        counts_ok &= static_cast<int>(ms.size()) > params.min_inliers;
        std::set<int> queries;
        for (const auto& m : ms) {
            indices_ok &= m.queryIdx >= 0 && m.queryIdx < static_cast<int>(fs.keypoints[i].size()) &&
                          m.trainIdx >= 0 && m.trainIdx < static_cast<int>(fs.keypoints[j].size());
            queries.insert(m.queryIdx);
        }
        unique_ok &= queries.size() == ms.size();  // each keypoint in i matched at most once
    }
    check(fs.keypoints.size() == set.images.size(), "keypoints for every image");
    check(!fs.matches.empty(), "some pairs verified");
    check(pairs_ok, "every pair has i < j within the window");
    check(counts_ok, "every pair has more than min_inliers matches");
    check(indices_ok, "all match indices point at real keypoints");
    check(unique_ok, "no keypoint in image i is matched twice within a pair");

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
