#pragma once
#include <string>
#include "lane/lane_types.hpp"

namespace lane {

struct Metrics {
    int frames = 0;
    int both_valid = 0;
    int left_only = 0;
    int right_only = 0;

    // simple running stats
    double lane_width_sum = 0.0;
    double lane_width_sq  = 0.0;
    int lane_width_n = 0;

    void update(const LaneEstimate& est, int frame_w, int frame_h);
    std::string summary() const;
};

} // namespace lane
