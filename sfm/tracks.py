"""Global feature tracks via Union-Find."""

import numpy as np


def build_global_tracks(num_images, keypoints, matches):
    """
    Build multi-view tracks from pairwise matches using Union-Find.

    Returns
    -------
    final_tracks : dict
        track_id -> list of (image_id, kp_index) with length >= 2
    img_kp_to_track : list[dict]
        img_kp_to_track[image_id][kp_index] = track_id
    """
    node_to_id = {}
    id_to_node = []
    for i in range(num_images):
        for k_idx in range(len(keypoints[i])):
            node_to_id[(i, k_idx)] = len(id_to_node)
            id_to_node.append((i, k_idx))

    num_nodes = len(id_to_node)
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=int)

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

    for (i, j), pair_matches in matches.items():
        for m in pair_matches:
            id_i = node_to_id[(i, m.queryIdx)]
            id_j = node_to_id[(j, m.trainIdx)]
            union(id_i, id_j)

    tracks = {}
    for node_idx in range(num_nodes):
        root = find(node_idx)
        if root not in tracks:
            tracks[root] = []
        tracks[root].append(id_to_node[node_idx])

    final_tracks = {tid: obs for tid, obs in tracks.items() if len(obs) >= 2}

    img_kp_to_track = [{} for _ in range(len(keypoints))]
    for track_id, observations in final_tracks.items():
        for img_id, kp_idx in observations:
            img_kp_to_track[img_id][kp_idx] = track_id

    print(f"Lookup table created. Ready to map {len(final_tracks)} global tracks.")
    print(f"Merged {len(matches)} pairs into {len(final_tracks)} global tracks.")
    return final_tracks, img_kp_to_track
