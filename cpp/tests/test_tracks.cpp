// Module 4 check: a hand-built case with a known answer, then invariants on the real data.
// Usage: test_tracks [data_dir [--no-refine] [--no-undistort] [--keep-conflicting]]
//        (no data_dir = synthetic case only)
#include <iostream>
#include <string>

#include "sfm/io.hpp"
#include "sfm/matching.hpp"
#include "sfm/tracks.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

cv::DMatch match(int q, int t) { return cv::DMatch(q, t, 0.f); }

// Every match joins one track (or, for dropped conflicting tracks, neither end has one),
// and the lookup table agrees with the track lists.
void check_invariants(const sfm::TrackSet& ts, const sfm::PairMatches& matches) {
    bool min_len = true, lookup = true, joined = true;
    size_t num_obs = 0;
    for (const auto& [tid, obs] : ts.tracks) {
        min_len &= obs.size() >= 2;
        num_obs += obs.size();
        for (const auto& [img, kp] : obs) {
            auto it = ts.kp_to_track[img].find(kp);
            lookup &= it != ts.kp_to_track[img].end() && it->second == tid;
        }
    }
    size_t lookup_size = 0;
    for (const auto& m : ts.kp_to_track) lookup_size += m.size();
    for (const auto& [pair, ms] : matches) {
        for (const auto& m : ms) {
            auto a = ts.kp_to_track[pair.first].find(m.queryIdx);
            auto b = ts.kp_to_track[pair.second].find(m.trainIdx);
            const bool has_a = a != ts.kp_to_track[pair.first].end();
            const bool has_b = b != ts.kp_to_track[pair.second].end();
            joined &= has_a == has_b && (!has_a || a->second == b->second);
        }
    }
    check(min_len, "every track has >= 2 observations");
    check(lookup && lookup_size == num_obs, "kp_to_track matches the track lists exactly");
    check(joined, "both ends of every match are in the same track");
}

}  // namespace

int main(int argc, char** argv) {
    // 3 images x 4 keypoints. Expected tracks:
    //   0: (0,0) (1,0) (2,0)
    //   1: (0,1) (1,1) (2,2) (0,3)   <- (0,3) joins via (0,2), so image 0 appears twice
    {
        sfm::Keypoints kps(3, std::vector<cv::KeyPoint>(4));
        sfm::PairMatches matches;
        matches[{0, 1}] = {match(0, 0), match(1, 1)};
        matches[{1, 2}] = {match(0, 0), match(1, 2)};
        matches[{0, 2}] = {match(3, 2)};

        const sfm::TrackSet ts = sfm::build_global_tracks(kps, matches, /*drop_conflicting=*/false);
        check(ts.tracks.size() == 2, "synthetic: 2 tracks");
        check(ts.tracks.count(0) && ts.tracks.at(0) == std::vector<sfm::Observation>{{0, 0}, {1, 0}, {2, 0}},
              "synthetic: track 0 is (0,0) (1,0) (2,0)");
        check(ts.tracks.count(1) && ts.tracks.at(1) == std::vector<sfm::Observation>{{0, 1}, {0, 3}, {1, 1}, {2, 2}},
              "synthetic: track 1 is (0,1) (0,3) (1,1) (2,2)");
        check(ts.num_conflicting == 1, "synthetic: 1 conflicting track");
        check(!ts.kp_to_track[0].count(2) && !ts.kp_to_track[2].count(1), "synthetic: unmatched keypoints have no track");
        check_invariants(ts, matches);

        // Dropping: the conflicting track disappears and its keypoints lose their track.
        const sfm::TrackSet td = sfm::build_global_tracks(kps, matches, /*drop_conflicting=*/true);
        check(td.tracks.size() == 1 && td.tracks.count(0) &&
                  td.tracks.at(0) == std::vector<sfm::Observation>{{0, 0}, {1, 0}, {2, 0}},
              "synthetic, drop: only track 0 remains");
        check(!td.kp_to_track[0].count(1) && !td.kp_to_track[0].count(3) && !td.kp_to_track[2].count(2),
              "synthetic, drop: conflicting keypoints have no track");
        check_invariants(td, matches);
    }

    if (argc > 1) {
        const std::string data_dir = argv[1];
        sfm::MatchingParams params;
        bool drop = true;
        for (int a = 2; a < argc; ++a) {
            if (std::string(argv[a]) == "--no-refine") params.refine_f = false;
            if (std::string(argv[a]) == "--no-undistort") params.undistort = false;
            if (std::string(argv[a]) == "--keep-conflicting") drop = false;
        }
        std::cout << "\nReal data: " << std::boolalpha << "refine_f = " << params.refine_f
                  << ", undistort = " << params.undistort << ", drop_conflicting = " << drop << "\n";
        const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
        const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);
        const sfm::FeatureSet fs = sfm::extract_features_and_matches(set.images, intr, params);
        const sfm::TrackSet ts = sfm::build_global_tracks(fs.keypoints, fs.matches, drop);

        std::map<size_t, int> length_hist;
        for (const auto& [tid, obs] : ts.tracks) ++length_hist[std::min<size_t>(obs.size(), 6)];
        std::cout << "Track length histogram:";
        for (const auto& [len, n] : length_hist) std::cout << "  " << (len == 6 ? "6+" : std::to_string(len)) << ": " << n;
        std::cout << "\n";
        check_invariants(ts, fs.matches);
    }

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
