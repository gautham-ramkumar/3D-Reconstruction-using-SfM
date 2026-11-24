#include "sfm/io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace sfm {

namespace {

bool has_png_extension(const std::string& name) {
    if (name.size() < 4) return false;
    std::string ext = name.substr(name.size() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext == ".png";
}

}  // namespace

ImageSet load_images(const std::string& image_dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        const std::string name = entry.path().filename().string();
        if (has_png_extension(name)) files.push_back(name);
    }
    std::sort(files.begin(), files.end());

    ImageSet set;
    for (const auto& name : files) {
        const std::string path = (fs::path(image_dir) / name).string();
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Warning: Failed to read " << path << "\n";
            continue;
        }
        set.images.push_back(img);
        set.names.push_back(name);
    }
    return set;
}

IntrinsicsMap load_intrinsics(const std::string& path, bool verbose) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open intrinsics file: " + path);

    IntrinsicsMap intrinsics;
    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '#') continue;

        std::istringstream ss(line);
        std::vector<std::string> tokens;
        for (std::string tok; ss >> tok;) tokens.push_back(tok);
        if (tokens.size() < 8) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": expected at least 8 fields, got " +
                                     std::to_string(tokens.size()));
        }

        const int cam_id = std::stoi(tokens[0]);
        const std::string& model = tokens[1];
        const double f = std::stod(tokens[4]);
        const double cx = std::stod(tokens[5]);
        const double cy = std::stod(tokens[6]);

        if (model != "SIMPLE_RADIAL") {
            std::cerr << "Warning: camera " << cam_id << " uses model " << model
                      << "; parsing it as SIMPLE_RADIAL\n";
        }

        Intrinsics intr;
        intr.width = std::stoi(tokens[2]);
        intr.height = std::stoi(tokens[3]);
        intr.k1 = std::stod(tokens[7]);
        intr.K << f, 0.0, cx,
                  0.0, f, cy,
                  0.0, 0.0, 1.0;

        if (verbose) {
            std::cout << "Camera ID: " << cam_id << "\n"
                      << "Focal Length: " << f << "\n"
                      << "Principal Point: (" << cx << ", " << cy << ")\n"
                      << "Radial Distortion (k1): " << intr.k1 << "\n"
                      << "Intrinsic Matrix K:\n" << intr.K << "\n\n";
        }
        intrinsics[cam_id] = intr;
    }
    return intrinsics;
}

}  // namespace sfm
