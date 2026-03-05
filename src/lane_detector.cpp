#include "lane/lane_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace lane {

namespace {

// helper finds the slope of the line for later filtering 
// Slope in image coords: left lane negative, right lane positive
static inline float segmentSlope(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    if (std::fabs(dx) < 1e-6f) return 1e6f;
    return dy / dx; // y = m*x + b
}

} // namespace

LaneDetector::LaneDetector(LaneConfig cfg) : cfg_(std::move(cfg)) {}

void LaneDetector::reset() {
    state_ = LaneEstimate{};
    last_update_frame_ = -1;
}

cv::Point2f LaneDetector::intersectY(const LaneLine& L, float y) {
    if (std::fabs(L.m) < 1e-6f) return cv::Point2f(L.b, y);
    float x = (y - L.b) / (L.m + 1e-6f);
    return cv::Point2f(x, y);
}

void LaneDetector::emaUpdate(LaneLine& prev, const LaneLine& cur, float alpha) {
    if (!cur.valid) return;
    if (!prev.valid) {
        prev = cur;
        return;
    }
    prev.m = prev.m * (1.f - alpha) + cur.m * alpha;
    prev.b = prev.b * (1.f - alpha) + cur.b * alpha;
    prev.valid = true;
}

// this is static, so config is passed in
LaneLine LaneDetector::fitLineToSide(const std::vector<cv::Point2f>& pts,
                                     const cv::Point& roiOffset,
                                     bool want_left,
                                     const LaneConfig& cfg) {
    LaneLine out;
    if (pts.size() < 6) return out; // need a few points for stability

    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 1e-2, 1e-2);

    float vx = line[0], vy = line[1];
    float x0 = line[2], y0 = line[3];

    // Convert ROI coords to   full frame coords
    x0 += roiOffset.x;
    y0 += roiOffset.y;

    if (std::fabs(vx) < 1e-6f) return out;
    float m = vy / vx;
    float b = y0 - x0 * m;

    bool is_left = (m < 0.f);
    if (is_left != want_left) return out;

    // Reject out of range slopes
    if (std::fabs(m) < cfg.slope_min || std::fabs(m) > cfg.slope_max) return out;

    out.valid = true;
    out.m = m;
    out.b = b;
    return out;
}

bool LaneDetector::laneWidthGateOk(const cv::Mat& frame,
                                   const LaneLine& L,
                                   const LaneLine& R) const {
    if (!L.valid || !R.valid) return false;

    const int H = frame.rows;
    const int W = frame.cols;

    float y = static_cast<float>(H - 1);
    cv::Point2f pl = intersectY(L, y);
    cv::Point2f pr = intersectY(R, y);

    float width = std::fabs(pr.x - pl.x);
    float min_w = cfg_.lane_width_min_frac_w * W;
    float max_w = cfg_.lane_width_max_frac_w * W;

    return (width >= min_w && width <= max_w);
}

LaneEstimate LaneDetector::update(const cv::Mat& frame_bgr, int frame_idx,
                                  const DetectionStageToggles& toggles,
                                  DetectionTimings* out_timings) {
    if (frame_bgr.empty()) return state_;

    // Only run full pipeline every N frames; otherwise return cached state
    const bool should_run_pipeline = (cfg_.update_every_n <= 1) ||
        (frame_idx - last_update_frame_ >= cfg_.update_every_n);
    if (!should_run_pipeline) {
        if (out_timings) *out_timings = DetectionTimings{};
        return state_;
    }

    const int H = frame_bgr.rows;
    const int W = frame_bgr.cols;
    const double freq = cv::getTickFrequency() * 0.001; // ms per tick

    cv::Rect roi;
    cv::Point roiOffset(0, 0);
    cv::Mat roi_img, gray, edges;
    std::vector<cv::Vec4i> segments;
    std::vector<cv::Point2f> left_pts, right_pts;

    // taking suitable ROI -- coords are set in lane types header 
    if (!toggles.roi) return state_;
    {
        int64_t t0 = cv::getTickCount();
        int x0 = std::clamp((int)std::round(cfg_.roi_x0 * W), 0, W - 2);
        int x1 = std::clamp((int)std::round(cfg_.roi_x1 * W), 1, W - 1);
        int y0 = std::clamp((int)std::round(cfg_.roi_y0 * H), 0, H - 2);
        int y1 = std::clamp((int)std::round(cfg_.roi_y1 * H), 1, H - 1);
        if (x1 <= x0 + 1) x1 = std::min(W - 1, x0 + 2);
        if (y1 <= y0 + 1) y1 = std::min(H - 1, y0 + 2);
        roi = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        roiOffset = cv::Point(x0, y0);
        roi_img = frame_bgr(roi);
        if (out_timings) out_timings->roi_ms = (cv::getTickCount() - t0) / freq;
    }

    // have to greyscale 
    if (!toggles.grayscale) return state_;
    {
        int64_t t0 = cv::getTickCount();
        cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
        if (out_timings) out_timings->grayscale_ms = (cv::getTickCount() - t0) / freq;
    }

    // blur it then, more textured roads will require a heavier blur 
    if (!toggles.blur) return state_;
    {
        int64_t t0 = cv::getTickCount();
        int k = (cfg_.blur_ksize % 2) ? cfg_.blur_ksize : (cfg_.blur_ksize + 1);
        if (k < 3) k = 3;
        cv::GaussianBlur(gray, gray, cv::Size(k, k), 0);
        if (out_timings) out_timings->blur_ms = (cv::getTickCount() - t0) / freq;
    }

    // canny edge detection using thresholds assigned in lane types
    if (!toggles.canny) return state_;
    {
        int64_t t0 = cv::getTickCount();
        cv::Canny(gray, edges, cfg_.canny_lo, cfg_.canny_hi);
        if (out_timings) out_timings->canny_ms = (cv::getTickCount() - t0) / freq;
    }

    // apply hough lines 
    if (!toggles.hough) return state_;
    {
        int64_t t0 = cv::getTickCount();
        float min_len = cfg_.min_line_len_frac_h * roi.height;
        cv::HoughLinesP(edges,
                        segments,
                        1,
                        CV_PI / 180,
                        cfg_.hough_threshold,
                        (double)min_len,
                        (double)cfg_.max_line_gap);
        if (out_timings) out_timings->hough_ms = (cv::getTickCount() - t0) / freq;
    }

    // sort points by segmenting left ot right 
    if (!toggles.sort_points) return state_;
    {
        int64_t t0 = cv::getTickCount();
        left_pts.clear();
        right_pts.clear();
        left_pts.reserve(segments.size() * 2);
        right_pts.reserve(segments.size() * 2);
        float anchor_y_min = cfg_.bottom_anchor_frac * roi.height;
        for (const auto& seg : segments) {
            float sx1 = (float)seg[0], sy1 = (float)seg[1];
            float sx2 = (float)seg[2], sy2 = (float)seg[3];
            float slope = segmentSlope(sx1, sy1, sx2, sy2);
            if (std::fabs(slope) < cfg_.slope_min || std::fabs(slope) > cfg_.slope_max)
                continue;
            if (std::max(sy1, sy2) < anchor_y_min)
                continue;
            if (slope < 0.f) {
                left_pts.emplace_back(sx1, sy1);
                left_pts.emplace_back(sx2, sy2);
            } else {
                right_pts.emplace_back(sx1, sy1);
                right_pts.emplace_back(sx2, sy2);
            }
        }
        if (out_timings) out_timings->sort_points_ms = (cv::getTickCount() - t0) / freq;
    }

    // fit lines, gates and do ema 

    if (!toggles.fit_and_gates) return state_;
    {
        int64_t t0 = cv::getTickCount();
        LaneLine left_line  = fitLineToSide(left_pts,  roiOffset, true,  cfg_);
        LaneLine right_line = fitLineToSide(right_pts, roiOffset, false, cfg_);

        if (state_.left.valid && left_line.valid &&
            std::fabs(left_line.m - state_.left.m) > cfg_.max_slope_jump)
            left_line.valid = false;
        if (state_.right.valid && right_line.valid &&
            std::fabs(right_line.m - state_.right.m) > cfg_.max_slope_jump)
            right_line.valid = false;

        if (left_line.valid && right_line.valid && !laneWidthGateOk(frame_bgr, left_line, right_line)) {
            left_line.valid = false;
            right_line.valid = false;
        }

        bool do_update = (cfg_.update_every_n <= 1) ||
                         (frame_idx - last_update_frame_ >= cfg_.update_every_n);
        if (do_update) {
            emaUpdate(state_.left,  left_line,  cfg_.ema_alpha);
            emaUpdate(state_.right, right_line, cfg_.ema_alpha);
            last_update_frame_ = frame_idx;
        }
        if (out_timings) out_timings->fit_and_gates_ms = (cv::getTickCount() - t0) / freq;
    }

    return state_;
}

void LaneDetector::drawOverlay(cv::Mat& frame_bgr, const LaneEstimate& est,
                               bool fill_lane, bool draw_roi) const {
    const int H = frame_bgr.rows;
    const int W = frame_bgr.cols;

    if (draw_roi) {
        int x0 = (int)std::round(cfg_.roi_x0 * W);
        int x1 = (int)std::round(cfg_.roi_x1 * W);
        int y0 = (int)std::round(cfg_.roi_y0 * H);
        int y1 = (int)std::round(cfg_.roi_y1 * H);
        cv::rectangle(frame_bgr, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    }

    auto drawLine = [&](const LaneLine& L, const cv::Scalar& color) {
        if (!L.valid) return;
        float y_bottom = (float)(H - 1);
        float y_top = (float)(cfg_.roi_y0 * H);
        cv::Point2f p1 = intersectY(L, y_bottom);
        cv::Point2f p2 = intersectY(L, y_top);
        cv::line(frame_bgr, p1, p2, color, 2, cv::LINE_AA);
    };

    // left green, right red 
    drawLine(est.left,  cv::Scalar(0, 255, 0));
    drawLine(est.right, cv::Scalar(0, 0, 255));

    if (fill_lane && est.bothValid()) {
        float y_bottom = (float)(H - 1);
        float y_top = (float)(cfg_.roi_y0 * H);

        cv::Point2f bl = intersectY(est.left,  y_bottom);
        cv::Point2f br = intersectY(est.right, y_bottom);
        cv::Point2f tl = intersectY(est.left,  y_top);
        cv::Point2f tr = intersectY(est.right, y_top);

        cv::Point pts[4] = { bl, br, tr, tl };

        // Semi transparent orange so road markings remain visible
        const double fill_alpha = 0.35;
        cv::Mat overlay = frame_bgr.clone();
        cv::fillConvexPoly(overlay, pts, 4, cv::Scalar(0, 128, 255), cv::LINE_AA);
        cv::Mat mask = cv::Mat::zeros(frame_bgr.size(), CV_8U);
        cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255));
        for (int y = 0; y < frame_bgr.rows; ++y) {
            const uchar* m = mask.ptr<uchar>(y);
            cv::Vec3b* dst = frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* src = overlay.ptr<cv::Vec3b>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                if (m[x]) {
                    dst[x][0] = cv::saturate_cast<uchar>((1.0 - fill_alpha) * dst[x][0] + fill_alpha * src[x][0]);
                    dst[x][1] = cv::saturate_cast<uchar>((1.0 - fill_alpha) * dst[x][1] + fill_alpha * src[x][1]);
                    dst[x][2] = cv::saturate_cast<uchar>((1.0 - fill_alpha) * dst[x][2] + fill_alpha * src[x][2]);
                }
            }
        }
    }
}

} // namespace lane
