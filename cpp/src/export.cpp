#include "sfm/export.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>

#include "sfm/camera.hpp"

namespace fs = std::filesystem;

namespace sfm {

namespace {

// Pixel as the real camera recorded it (what COLMAP and the image files expect).
Eigen::Vector2d raw_pixel(const Keypoints& keypoints, const IntrinsicsMap& intrinsics, int image,
                          int kp, bool undistorted) {
    const cv::Point2f& pt = keypoints[image][kp].pt;
    const Eigen::Vector2d uv(pt.x, pt.y);
    return undistorted ? distort_pixel(uv, intrinsics.at(image + 1)) : uv;
}

std::array<int, 3> sample_rgb(const cv::Mat& img, const Eigen::Vector2d& uv) {
    const int x = std::clamp(static_cast<int>(std::lround(uv.x())), 0, img.cols - 1);
    const int y = std::clamp(static_cast<int>(std::lround(uv.y())), 0, img.rows - 1);
    const cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);
    return {bgr[2], bgr[1], bgr[0]};
}

// COLMAP quaternion (qw, qx, qy, qz), normalized and with qw >= 0.
Eigen::Vector4d rotation_to_quaternion(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1;
    return {q.w(), q.x(), q.y(), q.z()};
}

// First registered observation of each mapped track, image-major (same order as Python).
std::map<int, Observation> first_observations(const Reconstruction& rec, const KpToTrack& kp_to_track) {
    std::map<int, Observation> first;
    for (const auto& [img, pose] : rec.poses) {
        std::vector<std::pair<int, int>> kps(kp_to_track[img].begin(), kp_to_track[img].end());
        std::sort(kps.begin(), kps.end());
        for (const auto& [kp, tid] : kps) {
            if (rec.points.count(tid) && !first.count(tid)) first[tid] = {img, kp};
        }
    }
    return first;
}

bool colmap_available() { return std::system("command -v colmap > /dev/null 2>&1") == 0; }

void convert_to_binary(const fs::path& dir) {
    if (!colmap_available()) {
        std::cout << "colmap not found on PATH; skipped binary conversion.\n";
        return;
    }
    // COLMAP prefers .bin over .txt, so stale binaries would hide the new text model.
    for (const char* name : {"cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"}) {
        fs::remove(dir / name);
    }
    const std::string cmd = "colmap model_converter --input_path \"" + dir.string() +
                            "\" --output_path \"" + dir.string() +
                            "\" --output_type BIN > /dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0) {
        std::cout << "Also wrote binary model (.bin) in " << dir.string() << "\n";
    } else {
        std::cout << "Binary conversion failed (command: " << cmd << ")\n";
    }
}

}  // namespace

std::string export_colmap_model(const std::string& output_dir, const Reconstruction& rec,
                                const KpToTrack& kp_to_track, const Keypoints& keypoints,
                                const IntrinsicsMap& intrinsics, const ImageSet& images,
                                bool undistorted, bool also_binary) {
    const fs::path dir = fs::absolute(output_dir);
    fs::create_directories(dir);

    std::map<int, int> tid_to_pid;
    for (const auto& kv : rec.points) tid_to_pid[kv.first] = static_cast<int>(tid_to_pid.size()) + 1;

    // 2D observations per image, and the (image_id, point2d_idx) track of each 3D point.
    std::map<int, std::vector<std::pair<Eigen::Vector2d, int>>> image_obs;
    std::map<int, std::vector<std::pair<int, int>>> point_tracks;
    for (const auto& [img, pose] : rec.poses) {
        auto& obs = image_obs[img];
        std::vector<std::pair<int, int>> kps(kp_to_track[img].begin(), kp_to_track[img].end());
        std::sort(kps.begin(), kps.end());
        for (const auto& [kp, tid] : kps) {
            const auto it = tid_to_pid.find(tid);
            if (it == tid_to_pid.end()) continue;
            point_tracks[it->second].emplace_back(img + 1, static_cast<int>(obs.size()));
            obs.emplace_back(raw_pixel(keypoints, intrinsics, img, kp, undistorted), it->second);
        }
    }

    std::ofstream cams(dir / "cameras.txt");
    cams << std::setprecision(17);
    cams << "# Camera list with one line of data per camera:\n"
         << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
         << "# Number of cameras: " << rec.poses.size() << "\n";
    for (const auto& [img, pose] : rec.poses) {
        const Intrinsics& in = intrinsics.at(img + 1);
        cams << img + 1 << " SIMPLE_RADIAL " << images.images[img].cols << " " << images.images[img].rows
             << " " << in.K(0, 0) << " " << in.K(0, 2) << " " << in.K(1, 2) << " " << in.k1 << "\n";
    }

    double mean_obs = 0;
    for (const auto& kv : image_obs) mean_obs += kv.second.size();
    mean_obs /= std::max<size_t>(image_obs.size(), 1);
    std::ofstream imgs(dir / "images.txt");
    imgs << std::setprecision(17);
    imgs << "# Image list with two lines of data per image:\n"
         << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
         << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
         << "# Number of images: " << rec.poses.size() << ", mean observations per image: " << mean_obs << "\n";
    for (const auto& [img, pose] : rec.poses) {
        const Eigen::Vector4d q = rotation_to_quaternion(pose.R);
        imgs << img + 1 << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << " " << pose.t.x()
             << " " << pose.t.y() << " " << pose.t.z() << " " << img + 1 << " " << images.names[img] << "\n";
        bool first = true;
        for (const auto& [uv, pid] : image_obs[img]) {
            imgs << (first ? "" : " ") << uv.x() << " " << uv.y() << " " << pid;
            first = false;
        }
        imgs << "\n";
    }

    const auto first_obs = first_observations(rec, kp_to_track);
    double mean_track = 0;
    for (const auto& kv : point_tracks) mean_track += kv.second.size();
    mean_track /= std::max<size_t>(point_tracks.size(), 1);
    std::ofstream pts(dir / "points3D.txt");
    pts << std::setprecision(17);
    pts << "# 3D point list with one line of data per point:\n"
        << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        << "# Number of points: " << rec.points.size() << ", mean track length: " << mean_track << "\n";
    for (const auto& [tid, X] : rec.points) {
        const int pid = tid_to_pid.at(tid);
        const auto& track = point_tracks[pid];

        // Mean reprojection error, computed in undistorted (pinhole) space.
        double err_sum = 0;
        int err_n = 0;
        for (const auto& [image_id, p2d] : track) {
            const int img = image_id - 1;
            const Pose& pose = rec.poses.at(img);
            const Eigen::Vector3d Xc = pose.R * X + pose.t;
            if (!(Xc.z() > 1e-8)) continue;
            const Eigen::Vector2d raw = image_obs[img][p2d].first;
            const Eigen::Vector2d uv = undistorted ? undistort_pixel(raw, intrinsics.at(image_id)) : raw;
            err_sum += ((intrinsics.at(image_id).K * Xc).hnormalized() - uv).norm();
            ++err_n;
        }

        std::array<int, 3> rgb{0, 0, 0};
        const auto fo = first_obs.find(tid);
        if (fo != first_obs.end()) {
            const auto [img, kp] = fo->second;
            rgb = sample_rgb(images.images[img], raw_pixel(keypoints, intrinsics, img, kp, undistorted));
        }
        pts << pid << " " << X.x() << " " << X.y() << " " << X.z() << " " << rgb[0] << " " << rgb[1]
            << " " << rgb[2] << " " << (err_n ? err_sum / err_n : 0.0);
        for (const auto& [image_id, p2d] : track) pts << " " << image_id << " " << p2d;
        pts << "\n";
    }
    cams.close();
    imgs.close();
    pts.close();

    std::cout << "Wrote COLMAP text model to " << dir.string() << " (" << rec.poses.size() << " images, "
              << rec.points.size() << " points)\n";
    if (also_binary) convert_to_binary(dir);
    return dir.string();
}

void export_ply(const std::string& path, const Reconstruction& rec, const KpToTrack& kp_to_track,
                const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                const ImageSet& images, bool undistorted) {
    const fs::path out = fs::absolute(path);
    if (out.has_parent_path()) fs::create_directories(out.parent_path());
    const auto first_obs = first_observations(rec, kp_to_track);

    std::ofstream ply(out);
    if (!ply) throw std::runtime_error("Cannot write " + out.string());
    ply << "ply\nformat ascii 1.0\n"
        << "comment " << rec.points.size() << " map points, then " << rec.poses.size()
        << " camera centers (red)\n"
        << "element vertex " << rec.points.size() + rec.poses.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    ply << std::setprecision(9);
    for (const auto& [tid, X] : rec.points) {
        std::array<int, 3> rgb{40, 40, 40};  // dark grey if no registered observation
        const auto fo = first_obs.find(tid);
        if (fo != first_obs.end()) {
            const auto [img, kp] = fo->second;
            rgb = sample_rgb(images.images[img], raw_pixel(keypoints, intrinsics, img, kp, undistorted));
        }
        ply << X.x() << " " << X.y() << " " << X.z() << " " << rgb[0] << " " << rgb[1] << " " << rgb[2] << "\n";
    }
    for (const auto& [img, pose] : rec.poses) {
        const Eigen::Vector3d C = pose.center();
        ply << C.x() << " " << C.y() << " " << C.z() << " 255 0 0\n";
    }
    std::cout << "Wrote PLY to " << out.string() << " (" << rec.points.size() << " points, "
              << rec.poses.size() << " cameras)\n";
}

bool open_in_colmap_gui(const std::string& model_dir, const std::string& image_dir,
                        const std::string& database_path, int min_track_len, double max_error) {
    if (!colmap_available()) {
        std::cout << "colmap not found on PATH.\n";
        return false;
    }
    std::ostringstream cmd;
    cmd << "colmap gui --database_path \"" << fs::absolute(database_path).string() << "\" --image_path \""
        << fs::absolute(image_dir).string() << "\" --import_path \"" << fs::absolute(model_dir).string()
        << "\" --Render.min_track_len " << min_track_len << " --Render.max_error " << max_error
        << " > /dev/null 2>&1 &";
    std::cout << "Launching: " << cmd.str() << "\n"
              << "Tip: In COLMAP, open Render options and increase Point size for bolder landmarks.\n";
    return std::system(cmd.str().c_str()) == 0;
}

}  // namespace sfm
