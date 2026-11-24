// Module 1 check: load the dataset and verify images, intrinsics and RNG behave.
// Usage: test_io <data_dir>   (data_dir holds buddha_images/ and cameras.txt)
#include <algorithm>
#include <iostream>
#include <set>
#include <string>

#include "sfm/io.hpp"
#include "sfm/rng.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : ".";

    const sfm::ImageSet set = sfm::load_images(data_dir + "/buddha_images");
    std::cout << "Loaded " << set.images.size() << " images\n";
    check(!set.images.empty(), "at least one image loaded");
    check(set.images.size() == set.names.size(), "one name per image");
    check(std::is_sorted(set.names.begin(), set.names.end()), "names are sorted");

    const sfm::IntrinsicsMap intr = sfm::load_intrinsics(data_dir + "/cameras.txt", false);
    std::cout << "Loaded " << intr.size() << " cameras\n";
    check(intr.size() == set.images.size(), "one camera per image");

    // Image i must use camera i + 1 and match its declared size.
    bool sizes_match = true;
    for (size_t i = 0; i < set.images.size(); ++i) {
        auto it = intr.find(static_cast<int>(i) + 1);
        if (it == intr.end() || it->second.width != set.images[i].cols ||
            it->second.height != set.images[i].rows) {
            sizes_match = false;
            std::cout << "  mismatch at image " << i << " (" << set.names[i] << ")\n";
        }
    }
    check(sizes_match, "image sizes match cameras.txt width/height");

    if (!intr.empty()) {
        const auto& c1 = intr.begin()->second;
        std::cout << "Camera " << intr.begin()->first << " K:\n" << c1.K << "\nk1 = " << c1.k1 << "\n";
    }

    sfm::set_random_seed(0);
    const auto s1 = sfm::sample_without_replacement(100, 8);
    sfm::set_random_seed(0);
    const auto s2 = sfm::sample_without_replacement(100, 8);
    check(s1 == s2, "same seed gives same sample");
    check(std::set<int>(s1.begin(), s1.end()).size() == 8, "sample has 8 distinct indices");

    std::cout << (failures == 0 ? "All checks passed.\n" : "Some checks FAILED.\n");
    return failures == 0 ? 0 : 1;
}
