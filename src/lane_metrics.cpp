#include "lane/lane_metrics.hpp"
#include <opencv2/core.hpp>
#include <sstream>
#include <cmath>

namespace lane {

namespace {
float lineXatY(const LaneLine& L, float y) {
    if (std::fabs(L.m) < 1e-6f) return L.b;
    return (y - L.b) / L.m;
}
} // namespace


// helpers used for logging metrics on lane detector 
// more so used in the testing phases
// will not be printing all of this in main project as it runs 


void Metrics::update(const LaneEstimate& est, int frame_w, int frame_h) {
    (void)frame_w;
    ++frames;
    if (est.bothValid()) {
        ++both_valid;
        float bottom_y = static_cast<float>(frame_h - 1);
        float xl = lineXatY(est.left, bottom_y);
        float xr = lineXatY(est.right, bottom_y);
        double width = std::fabs(xr - xl);
        lane_width_sum += width;
        lane_width_sq += width * width;
        ++lane_width_n;
    } else if (est.left.valid) {
        ++left_only;
    } else if (est.right.valid) {
        ++right_only;
    }
}

std::string Metrics::summary() const {
    std::ostringstream os;
    os << "frames=" << frames
       << " both_valid=" << both_valid
       << " left_only=" << left_only
       << " right_only=" << right_only;
    if (lane_width_n > 0) {
        double mean = lane_width_sum / lane_width_n;
        double var = (lane_width_sq / lane_width_n) - (mean * mean);
        double std = var > 0 ? std::sqrt(var) : 0.0;
        os << " lane_width_mean=" << mean << " lane_width_std=" << std << " n=" << lane_width_n;
    }
    return os.str();
}

} // namespace lane
