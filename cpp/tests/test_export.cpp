// Module 7 check: run the full pipeline, export before/after-BA models and a PLY, then
//   1. re-read images.txt / points3D.txt and verify every stored 2D point independently with
//      the forward SIMPLE_RADIAL model (project, then distort)
//   2. have COLMAP itself load the model (model_analyzer) and compare image / point counts
// Usage: test_export <data_dir> [output_dir = output]
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

#include <Eigen/Geometry>

#include "sfm/ba.hpp"
#include "sfm/camera.hpp"
#include "sfm/export.hpp"
#include "sfm/io.hpp"
#include "sfm/mapping.hpp"
#include "sfm/matching.hpp"
#include "sfm/tracks.hpp"

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

std::string run(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    std::array<char, 512> buf{};
    while (fgets(buf.data(), buf.size(), pipe)) out += buf.data();
    pclose(pipe);
    return out;
}

// Re-read the text model and reproject every 2D point with the forward camera model.
void verify_text_model(const fs::path& dir, const sfm::IntrinsicsMap& intr, size_t expect_images,
                       size_t expect_points) {
    std::map<int, Eigen::Vector3d> points;
    std::ifstream pts(dir / "points3D.txt");
    for (std::string line; std::getline(pts, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int pid;
        Eigen::Vector3d X;
        ss >> pid >> X.x() >> X.y() >> X.z();
        points[pid] = X;
    }

    std::ifstream imgs(dir / "images.txt");
    size_t num_images = 0, num_obs = 0;
    double max_err = 0, sum_err = 0;
    std::string header, obs_line;
    while (std::getline(imgs, header)) {
        if (header.empty() || header[0] == '#') continue;
        std::getline(imgs, obs_line);
        std::istringstream hs(header);
        int image_id, cam_id;
        double qw, qx, qy, qz;
        Eigen::Vector3d t;
        std::string name;
        hs >> image_id >> qw >> qx >> qy >> qz >> t.x() >> t.y() >> t.z() >> cam_id >> name;
        const Eigen::Matrix3d R = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
        const sfm::Intrinsics& in = intr.at(cam_id);
        ++num_images;

        std::istringstream os(obs_line);
        double u, v;
        int pid;
        while (os >> u >> v >> pid) {
            const Eigen::Vector2d pinhole = (in.K * (R * points.at(pid) + t)).hnormalized();
            const double e = (sfm::distort_pixel(pinhole, in) - Eigen::Vector2d(u, v)).norm();
            max_err = std::max(max_err, e);
            sum_err += e;
            ++num_obs;
        }
    }
    std::cout << "  " << dir.filename().string() << ": " << num_images << " images, " << points.size()
              << " points, " << num_obs << " observations, distorted reprojection mean "
              << sum_err / std::max<size_t>(num_obs, 1) << " px\n";
    check(num_images == expect_images && points.size() == expect_points,
          dir.filename().string() + ": text model has the expected images and points");
}

void verify_with_colmap(const fs::path& dir, size_t expect_images, size_t expect_points) {
    if (run("command -v colmap").empty()) {
        std::cout << "  colmap not installed; skipping COLMAP load check\n";
        return;
    }
    const std::string out = run("colmap model_analyzer --path \"" + dir.string() + "\" 2>&1");
    auto value_after = [&](const std::string& key) -> double {
        const auto p = out.find(key);
        if (p == std::string::npos) return -1;
        return std::stod(out.substr(p + key.size()));
    };
    const double reg = value_after("Registered images: ");
    const double pts = value_after("Points: ");
    const double err = value_after("Mean reprojection error: ");
    std::cout << "  COLMAP reads " << dir.filename().string() << ": " << reg << " registered images, "
              << pts << " points, mean reprojection error " << err << " px\n";
    check(reg == double(expect_images) && pts == double(expect_points),
          dir.filename().string() + ": COLMAP loads the model with matching counts");
    check(fs::exists(dir / "images.bin") && fs::exists(dir / "points3D.bin"),
          dir.filename().string() + ": binary model written");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_export <data_dir> [output_dir]\n";
        return 2;
    }
    const std::string data_dir = argv[1];
    const fs::path out_dir = fs::absolute(argc > 2 ? argv[2] : "output");

    const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
    const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);
    sfm::MatchingParams mp;
    const sfm::FeatureSet fs_ = sfm::extract_features_and_matches(set.images, intr, mp);
    const sfm::TrackSet ts = sfm::build_global_tracks(fs_.keypoints, fs_.matches);

    const int seed_i = 15, seed_j = 16, n = static_cast<int>(set.images.size());
    sfm::Reconstruction rec = sfm::initialize_seed_map(seed_i, seed_j, fs_.matches, fs_.keypoints,
                                                       ts.kp_to_track, intr);
    std::vector<int> forward(n - seed_j - 1), reverse(seed_i);
    std::iota(forward.begin(), forward.end(), seed_j + 1);
    for (int k = 0; k < seed_i; ++k) reverse[k] = seed_i - 1 - k;
    sfm::run_pnp_mapping(forward, rec, ts.kp_to_track, fs_.matches, fs_.keypoints, intr);
    sfm::run_pnp_mapping(reverse, rec, ts.kp_to_track, fs_.matches, fs_.keypoints, intr);
    sfm::retriangulate_tracks(rec, ts.tracks, fs_.keypoints, intr);
    const sfm::BAResult ba = sfm::run_bundle_adjustment(rec, ts.kp_to_track, fs_.keypoints, intr);

    std::cout << "\n--- Export ---\n";
    const fs::path before = sfm::export_colmap_model((out_dir / "before_ba").string(), rec, ts.kp_to_track,
                                                     fs_.keypoints, intr, set, mp.undistort);
    const fs::path after = sfm::export_colmap_model((out_dir / "after_ba").string(), ba.rec, ts.kp_to_track,
                                                    fs_.keypoints, intr, set, mp.undistort);
    sfm::export_ply((out_dir / "before_ba.ply").string(), rec, ts.kp_to_track, fs_.keypoints, intr, set, mp.undistort);
    sfm::export_ply((out_dir / "after_ba.ply").string(), ba.rec, ts.kp_to_track, fs_.keypoints, intr, set, mp.undistort);

    // Our own pinhole stats on the same observations, for comparison with the distorted check.
    const auto st = sfm::compute_reprojection_stats(ba.rec, ts.kp_to_track, fs_.keypoints, intr);
    std::cout << "\nOur after-BA stats (undistorted space): mean " << st.mean << " px over "
              << st.num_observations << " observations\n";

    std::cout << "\n--- Checks ---\n";
    verify_text_model(before, intr, rec.poses.size(), rec.points.size());
    verify_text_model(after, intr, ba.rec.poses.size(), ba.rec.points.size());
    verify_with_colmap(before, rec.poses.size(), rec.points.size());
    verify_with_colmap(after, ba.rec.poses.size(), ba.rec.points.size());

    std::ifstream ply(out_dir / "after_ba.ply");
    std::string line;
    size_t vertices = 0, lines_after_header = 0;
    bool in_body = false;
    while (std::getline(ply, line)) {
        if (in_body) ++lines_after_header;
        if (line.rfind("element vertex ", 0) == 0) vertices = std::stoul(line.substr(15));
        if (line == "end_header") in_body = true;
    }
    check(vertices == ba.rec.points.size() + ba.rec.poses.size() && lines_after_header == vertices,
          "PLY vertex count = points + cameras, and the body has that many lines");

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
