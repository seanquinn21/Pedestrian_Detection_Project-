#pragma once
#include <opencv2/core.hpp>

namespace lane {

struct LaneLine {
    bool  valid = false;
    float m = 0.f;   // slope
    float b = 0.f;   // intercept: y = m*x + b
};

struct LaneEstimate {
    LaneLine left;
    LaneLine right;
    bool bothValid() const { return left.valid && right.valid; }
};

struct LaneConfig {
    // ROI rectangle (fractions of W and H) 
    float roi_y0 = 0.68f;  // start of ROI  as fraction of H
    float roi_y1 = 1.00f;
    float roi_x0 = 0.33f;
    float roi_x1 = 0.88f;  

    // Canny 
    int blur_ksize = 21;   // needs to be very high for textured roads 
    int canny_lo = 20;    // low threshold
    int canny_hi = 80;   // high threshold

    // Hough
    int   hough_threshold = 20;
    float min_line_len_frac_h = 0.05f; // fraction of ROI height
    int   max_line_gap = 120;

    // Filters
    float slope_min = 0.35f; // reject near horizontal
    float slope_max = 5.00f; // reject near vertical
    float bottom_anchor_frac = 0.75f; // line must touch bottom % of ROI

    // Temporal smoothing 
    float ema_alpha = 0.20f;
    int   update_every_n = 3;

    // Sanity gates (frame width based)
    float lane_width_min_frac_w = 0.35f;
    float lane_width_max_frac_w = 0.85f;
    float max_slope_jump = 1.0f; // reject big slope jumps
};

} // namespace lane
