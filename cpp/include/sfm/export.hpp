// Export reconstructions: COLMAP sparse model (for the COLMAP GUI) and a colored PLY.
#pragma once

#include <string>

#include "sfm/io.hpp"
#include "sfm/mapping.hpp"
#include "sfm/types.hpp"

namespace sfm {

// Write cameras.txt / images.txt / points3D.txt (and .bin via the colmap CLI if available).
// `undistorted` says whether the keypoints were undistorted; if so, 2D points are converted
// back to raw pixels so they match the SIMPLE_RADIAL camera model and the real images.
// Returns the absolute path of the model directory.
std::string export_colmap_model(const std::string& output_dir, const Reconstruction& rec,
                                const KpToTrack& kp_to_track, const Keypoints& keypoints,
                                const IntrinsicsMap& intrinsics, const ImageSet& images,
                                bool undistorted, bool also_binary = true);

// ASCII PLY: map points colored from their first observation, camera centers in red.
void export_ply(const std::string& path, const Reconstruction& rec, const KpToTrack& kp_to_track,
                const Keypoints& keypoints, const IntrinsicsMap& intrinsics,
                const ImageSet& images, bool undistorted);

// Launch the COLMAP GUI on a model in the background. Defaults show 2-view points (COLMAP's
// default min_track_len = 3 hides most of this pipeline's points). Returns false if colmap
// is not installed.
bool open_in_colmap_gui(const std::string& model_dir, const std::string& image_dir,
                        const std::string& database_path, int min_track_len = 2,
                        double max_error = 10.0);

}  // namespace sfm
