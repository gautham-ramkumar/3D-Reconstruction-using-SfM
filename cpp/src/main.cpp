// End-to-end incremental SfM, replacing SFM.ipynb:
// load -> match -> tracks -> seed -> PnP passes -> retriangulate -> export -> BA -> export.
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "sfm/ba.hpp"
#include "sfm/export.hpp"
#include "sfm/io.hpp"
#include "sfm/mapping.hpp"
#include "sfm/matching.hpp"
#include "sfm/rng.hpp"
#include "sfm/tracks.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string data_dir = ".";
    std::string out_dir;  // default: <data_dir>/colmap_exports_cpp
    int seed_i = 15, seed_j = 16;  // the notebook's proven seed pair
    bool auto_seed = false;
    bool undistort = true;
    bool refine_f = true;
    bool drop_conflicting = true;
    bool binary = true;
    double min_tri_angle = sfm::kDefaultMinTriAngleDeg;
    std::string gui;  // "", "before" or "after"
};

void usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --data-dir D        folder with buddha_images/ and cameras.txt (default .)\n"
        << "  --out-dir O         output folder (default D/colmap_exports_cpp)\n"
        << "  --seed I J          seed image pair (default 15 16)\n"
        << "  --auto-seed         pick the seed with find_best_seed instead\n"
        << "  --min-tri-angle A   minimum triangulation angle in degrees (default 1.5, 0 = off)\n"
        << "  --no-undistort      keep raw (distorted) keypoint positions\n"
        << "  --no-refine         do not refit F on RANSAC inliers\n"
        << "  --keep-conflicting  keep tracks with 2+ keypoints in one image\n"
        << "  --no-binary         skip the COLMAP .bin conversion\n"
        << "  --gui before|after  open that model in the COLMAP GUI at the end\n";
}

Options parse(int argc, char** argv) {
    Options o;
    auto need = [&](int& a, int n) {
        if (a + n >= argc) {
            std::cerr << "Missing value for " << argv[a] << "\n";
            std::exit(2);
        }
    };
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--data-dir") { need(a, 1); o.data_dir = argv[++a]; }
        else if (arg == "--out-dir") { need(a, 1); o.out_dir = argv[++a]; }
        else if (arg == "--seed") { need(a, 2); o.seed_i = std::stoi(argv[++a]); o.seed_j = std::stoi(argv[++a]); }
        else if (arg == "--auto-seed") o.auto_seed = true;
        else if (arg == "--min-tri-angle") { need(a, 1); o.min_tri_angle = std::stod(argv[++a]); }
        else if (arg == "--no-undistort") o.undistort = false;
        else if (arg == "--no-refine") o.refine_f = false;
        else if (arg == "--keep-conflicting") o.drop_conflicting = false;
        else if (arg == "--no-binary") o.binary = false;
        else if (arg == "--gui") { need(a, 1); o.gui = argv[++a]; }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown option " << arg << "\n"; usage(argv[0]); std::exit(2); }
    }
    if (o.out_dir.empty()) o.out_dir = (fs::path(o.data_dir) / "colmap_exports_cpp").string();
    if (o.seed_i > o.seed_j) std::swap(o.seed_i, o.seed_j);  // matches are keyed with i < j
    return o;
}

class StageTimer {
public:
    void start(const std::string& name) {
        name_ = name;
        t0_ = std::chrono::steady_clock::now();
        std::cout << "\n===== " << name << " =====\n";
    }
    void stop() {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        times_.emplace_back(name_, s);
    }
    void report() const {
        std::cout << "\nTime per stage:\n";
        double total = 0;
        for (const auto& [n, s] : times_) {
            std::cout << "  " << std::left << std::setw(26) << n << std::right << std::fixed
                      << std::setprecision(2) << std::setw(7) << s << " s\n";
            total += s;
        }
        std::cout << "  " << std::left << std::setw(26) << "total" << std::right << std::setw(7) << total
                  << " s\n" << std::defaultfloat;
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
    std::vector<std::pair<std::string, double>> times_;
};

void print_stats(const char* label, const sfm::Reconstruction& rec, const sfm::ReprojectionStats& s) {
    std::cout << "  " << std::left << std::setw(10) << label << std::right << rec.poses.size()
              << " cameras, " << rec.points.size() << " points, " << s.num_observations
              << " observations | reprojection median " << std::setprecision(3) << s.median << " px, p90 "
              << s.p90 << " px, mean " << s.mean << " px\n" << std::defaultfloat;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    StageTimer timer;
    sfm::set_random_seed(0);

    timer.start("Load data");
    const sfm::ImageSet set = sfm::load_images((fs::path(opt.data_dir) / "buddha_images").string());
    const sfm::IntrinsicsMap intr = sfm::load_intrinsics((fs::path(opt.data_dir) / "cameras.txt").string(), false);
    const int n = static_cast<int>(set.images.size());
    std::cout << "Loaded " << n << " images and " << intr.size() << " cameras\n";
    if (n == 0) return 1;
    timer.stop();

    timer.start("Features and matching");
    sfm::MatchingParams mp;
    mp.undistort = opt.undistort;
    mp.refine_f = opt.refine_f;
    const sfm::FeatureSet fs_ = sfm::extract_features_and_matches(set.images, intr, mp);
    timer.stop();

    timer.start("Tracks");
    const sfm::TrackSet ts = sfm::build_global_tracks(fs_.keypoints, fs_.matches, opt.drop_conflicting);
    timer.stop();

    timer.start("Seed");
    int seed_i = opt.seed_i, seed_j = opt.seed_j;
    if (opt.auto_seed) {
        const auto best = sfm::find_best_seed(fs_.matches, intr, fs_.keypoints, 15, 2.0, 5000, opt.refine_f);
        if (!best) {
            std::cerr << "No seed pair has enough parallax; pass --seed I J\n";
            return 1;
        }
        std::tie(seed_i, seed_j) = best->pair;
        std::cout << "Auto seed: (" << seed_i << ", " << seed_j << ")\n";
    }
    if (!fs_.matches.count({seed_i, seed_j})) {
        std::cerr << "Seed pair (" << seed_i << ", " << seed_j << ") has no verified matches\n";
        return 1;
    }
    sfm::Reconstruction rec = sfm::initialize_seed_map(seed_i, seed_j, fs_.matches, fs_.keypoints, ts.kp_to_track,
                                                       intr, 0.1, 5000, opt.refine_f, opt.min_tri_angle);
    timer.stop();

    timer.start("PnP registration");
    sfm::PnPParams pnp;
    pnp.min_tri_angle_deg = opt.min_tri_angle;
    // Forward from seed_i + 1 so a non-adjacent seed does not skip the images between them.
    std::vector<int> forward(n - seed_i - 1), reverse(seed_i), catch_up(n);
    std::iota(forward.begin(), forward.end(), seed_i + 1);
    for (int k = 0; k < seed_i; ++k) reverse[k] = seed_i - 1 - k;
    std::iota(catch_up.begin(), catch_up.end(), 0);
    std::cout << "Starting Forward Pass\n";
    sfm::run_pnp_mapping(forward, rec, ts.kp_to_track, fs_.matches, fs_.keypoints, intr, pnp);
    std::cout << "\nStarting Reverse Pass\n";
    sfm::run_pnp_mapping(reverse, rec, ts.kp_to_track, fs_.matches, fs_.keypoints, intr, pnp);
    if (static_cast<int>(rec.poses.size()) < n) {
        std::cout << "\nStarting Catch-up Pass\n";
        sfm::run_pnp_mapping(catch_up, rec, ts.kp_to_track, fs_.matches, fs_.keypoints, intr, pnp);
    }
    std::cout << "\nMap after PnP: " << rec.points.size() << " points\n";
    sfm::retriangulate_tracks(rec, ts.tracks, fs_.keypoints, intr, 0.1, 4.0, opt.min_tri_angle);
    timer.stop();

    timer.start("Export before BA");
    sfm::diagnose_map_quality(rec, ts.kp_to_track, fs_.keypoints);
    const auto before = sfm::compute_reprojection_stats(rec, ts.kp_to_track, fs_.keypoints, intr);
    const std::string before_dir = sfm::export_colmap_model((fs::path(opt.out_dir) / "before_ba").string(), rec,
                                                            ts.kp_to_track, fs_.keypoints, intr, set,
                                                            opt.undistort, opt.binary);
    sfm::export_ply((fs::path(opt.out_dir) / "before_ba.ply").string(), rec, ts.kp_to_track, fs_.keypoints, intr,
                    set, opt.undistort);
    timer.stop();

    timer.start("Bundle adjustment");
    sfm::BAParams bp;
    bp.anchor_image = seed_i;  // seed_i defines the world frame
    bp.scale_image = seed_j;
    const sfm::BAResult ba = sfm::run_bundle_adjustment(rec, ts.kp_to_track, fs_.keypoints, intr, bp);
    timer.stop();

    timer.start("Export after BA");
    sfm::diagnose_map_quality(ba.rec, ts.kp_to_track, fs_.keypoints);
    const auto after = sfm::compute_reprojection_stats(ba.rec, ts.kp_to_track, fs_.keypoints, intr);
    const std::string after_dir = sfm::export_colmap_model((fs::path(opt.out_dir) / "after_ba").string(), ba.rec,
                                                           ts.kp_to_track, fs_.keypoints, intr, set,
                                                           opt.undistort, opt.binary);
    sfm::export_ply((fs::path(opt.out_dir) / "after_ba.ply").string(), ba.rec, ts.kp_to_track, fs_.keypoints,
                    intr, set, opt.undistort);
    timer.stop();

    std::cout << "\n===== Summary =====\n"
              << "  seed (" << seed_i << ", " << seed_j << "), undistort " << std::boolalpha << opt.undistort
              << ", refine F " << opt.refine_f << ", drop conflicting " << opt.drop_conflicting
              << ", min triangulation angle " << opt.min_tri_angle << " deg\n";
    print_stats("before BA", rec, before);
    print_stats("after BA", ba.rec, after);
    const auto& last = ba.rounds.back();
    std::cout << "  BA: " << last.num_observations << " observations in the final round, Python-style RMSE "
              << last.python_rmse << " px, plain RMS " << last.plain_rms << " px, round-1 iterations "
              << ba.rounds.front().iterations << "\n";
    timer.report();

    const std::string db = (fs::path(opt.data_dir) / "buddha_images" / "database.db").string();
    const std::string img_dir = (fs::path(opt.data_dir) / "buddha_images").string();
    if (!opt.gui.empty()) {
        sfm::open_in_colmap_gui(opt.gui == "before" ? before_dir : after_dir, img_dir, db);
    } else {
        std::cout << "\nView in COLMAP:\n  colmap gui --database_path " << fs::absolute(db).string()
                  << " --image_path " << fs::absolute(img_dir).string() << " --import_path " << after_dir
                  << " --Render.min_track_len 2 --Render.max_error 10\n";
    }
    return 0;
}
