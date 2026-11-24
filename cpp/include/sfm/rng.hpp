// Reproducibility helpers.
#pragma once

#include <random>
#include <vector>

namespace sfm {

// Shared generator for all RANSAC sampling in the pipeline.
std::mt19937& rng();

// Seed our generator and OpenCV's; force OpenCV to one thread for repeatable runs.
void set_random_seed(unsigned int seed = 0);

// k distinct indices from [0, n), via a partial Fisher-Yates shuffle.
std::vector<int> sample_without_replacement(int n, int k);

}  // namespace sfm
