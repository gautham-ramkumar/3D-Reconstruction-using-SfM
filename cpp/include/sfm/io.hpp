// Data loading helpers for images and COLMAP intrinsics.
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "sfm/types.hpp"

namespace sfm {

struct ImageSet {
    std::vector<cv::Mat> images;     // BGR, in sorted filename order
    std::vector<std::string> names;  // filename of each loaded image (same order)
};

// Load PNG images from a directory in sorted filename order.
ImageSet load_images(const std::string& image_dir);

// Parse COLMAP cameras.txt (SIMPLE_RADIAL model).
// Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, f, cx, cy, k1
IntrinsicsMap load_intrinsics(const std::string& path, bool verbose = true);

}  // namespace sfm
