#include "sfm/tracks.hpp"

#include <iostream>
#include <numeric>
#include <set>

namespace sfm {

namespace {

class UnionFind {
public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    // Iterative with path compression; recursion could overflow on long chains.
    int find(int v) {
        int root = v;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[v] != root) {
            const int next = parent_[v];
            parent_[v] = root;
            v = next;
        }
        return root;
    }

    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank_[ra] < rank_[rb]) {
            parent_[ra] = rb;
        } else if (rank_[ra] > rank_[rb]) {
            parent_[rb] = ra;
        } else {
            parent_[rb] = ra;
            ++rank_[ra];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

}  // namespace

TrackSet build_global_tracks(const Keypoints& keypoints, const PairMatches& matches,
                             bool drop_conflicting) {
    const int num_images = static_cast<int>(keypoints.size());

    // node(i, k) = offset[i] + k gives every keypoint a global ID.
    std::vector<int> offset(num_images + 1, 0);
    for (int i = 0; i < num_images; ++i) {
        offset[i + 1] = offset[i] + static_cast<int>(keypoints[i].size());
    }
    const int num_nodes = offset[num_images];

    UnionFind uf(num_nodes);
    for (const auto& [pair, pair_matches] : matches) {
        const auto [i, j] = pair;
        for (const auto& m : pair_matches) {
            uf.unite(offset[i] + m.queryIdx, offset[j] + m.trainIdx);
        }
    }

    // Group by root. Visiting nodes in increasing order keeps first-appearance order.
    std::vector<int> root_to_group(num_nodes, -1);
    std::vector<std::vector<Observation>> groups;
    for (int img = 0; img < num_images; ++img) {
        for (int k = 0; k < static_cast<int>(keypoints[img].size()); ++k) {
            const int root = uf.find(offset[img] + k);
            if (root_to_group[root] < 0) {
                root_to_group[root] = static_cast<int>(groups.size());
                groups.emplace_back();
            }
            groups[root_to_group[root]].emplace_back(img, k);
        }
    }

    TrackSet out;
    out.kp_to_track.resize(num_images);
    for (auto& obs : groups) {
        if (obs.size() < 2) continue;

        std::set<int> images_seen;
        bool conflict = false;
        for (const auto& obs_i : obs) conflict |= !images_seen.insert(obs_i.first).second;
        out.num_conflicting += conflict;
        if (conflict && drop_conflicting) continue;

        const int track_id = static_cast<int>(out.tracks.size());
        for (const auto& [img, kp] : obs) out.kp_to_track[img][kp] = track_id;
        out.tracks[track_id] = std::move(obs);
    }

    std::cout << "Lookup table created. Ready to map " << out.tracks.size() << " global tracks.\n"
              << "Merged " << matches.size() << " pairs into " << out.tracks.size()
              << " global tracks.\n"
              << "Conflicting tracks (2+ keypoints in one image): " << out.num_conflicting
              << (drop_conflicting ? " (dropped)" : " (kept)") << "\n";
    return out;
}

}  // namespace sfm
