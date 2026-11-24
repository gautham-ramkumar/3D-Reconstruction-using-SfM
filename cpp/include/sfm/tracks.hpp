// Global feature tracks via union-find.
#pragma once

#include "sfm/types.hpp"

namespace sfm {

struct TrackSet {
    Tracks tracks;            // track_id -> observations, every track has >= 2
    KpToTrack kp_to_track;    // kp_to_track[image][kp_idx] = track_id
    int num_conflicting = 0;  // tracks with 2+ different keypoints in one image
};

// Merge pairwise matches into multi-view tracks. Track IDs are 0, 1, 2, ... in order of
// each track's first keypoint (image-major), matching the Python grouping order.
// With drop_conflicting, conflicting tracks are left out entirely (their keypoints get no
// track), so later stages must skip matches whose keypoints have no track.
TrackSet build_global_tracks(const Keypoints& keypoints, const PairMatches& matches,
                             bool drop_conflicting = true);

}  // namespace sfm
