#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "lane/lane_types.hpp"

namespace lane {

// used in testing to check where bottlenecks lie in the detections pipeline 
struct DetectionStageToggles {
    bool roi          = true;  // extract ROI crop
    bool grayscale    = true;  // cvtColor BGR to grey 
    bool blur         = true;  // GaussianBlur
    bool canny        = true;  // canny edge detection
    bool hough        = true;  // HoughLinesP
    bool sort_points  = true;  // segment to   left/right point lists
    bool fit_and_gates = true; // fit lines, slope , EMA
};

// timings for each stage 
struct DetectionTimings {
    double roi_ms          = 0.0;
    double grayscale_ms    = 0.0;
    double blur_ms         = 0.0;
    double canny_ms        = 0.0;
    double hough_ms        = 0.0;
    double sort_points_ms  = 0.0;
    double fit_and_gates_ms = 0.0;
};

class LaneDetector {
public:
    explicit LaneDetector(LaneConfig cfg = {});
    void reset();

    // call once per frame..  toggles control which sub-stages run --  out_timings receives per-step ms when non-null
    LaneEstimate update(const cv::Mat& frame_bgr, int frame_idx,
                        const DetectionStageToggles& toggles = {},
                        DetectionTimings* out_timings = nullptr);

    // utility drawing
    void drawOverlay(cv::Mat& frame_bgr, const LaneEstimate& est,
                     bool fill_lane = true, bool draw_roi = false) const;

private:
    LaneConfig cfg_;
    LaneEstimate state_;
    int last_update_frame_ = -1;

    // helpers
	
    static LaneLine fitLineToSide(const std::vector<cv::Point2f>& pts,
                              const cv::Point& roiOffset,
                              bool want_left,
                              const LaneConfig& cfg);
    
    static void emaUpdate(LaneLine& prev, const LaneLine& cur, float alpha);
    static cv::Point2f intersectY(const LaneLine& L, float y);
    bool laneWidthGateOk(const cv::Mat& frame, const LaneLine& L, const LaneLine& R) const;
};

} // namespace lane
