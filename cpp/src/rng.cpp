#include "sfm/rng.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace sfm {

std::mt19937& rng() {
    static std::mt19937 generator(0);
    return generator;
}

void set_random_seed(unsigned int seed) {
    rng().seed(seed);
    cv::setRNGSeed(static_cast<int>(seed));  // used by solvePnPRansac
    // SIFT / matchers can be nondeterministic with multi-threading on some builds
    cv::setNumThreads(1);
    std::cout << "Random seed set to " << seed << " (std::mt19937, OpenCV; OpenCV threads=1)\n";
}

std::vector<int> sample_without_replacement(int n, int k) {
    if (k > n || k < 0) {
        throw std::invalid_argument("sample_without_replacement: need 0 <= k <= n");
    }
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (int a = 0; a < k; ++a) {
        std::uniform_int_distribution<int> pick(a, n - 1);
        std::swap(idx[a], idx[pick(rng())]);
    }
    idx.resize(k);
    return idx;
}

}  // namespace sfm
