// main application -- captures framaes from cam , runs YOLOv8 TRT engine 
// decode detections , runs NMS , draw boxes , logs FPS and persons count 

// added now: lidar fusion - computes the angle of where box is in cam fov and maps that to lidar angle and tags it with its distance 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include "lidar_c1.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/Rect.h"
#include "ByteTrack/STrack.h"

using namespace cv; using namespace std;

// TRT wrappers from trt_yolo.cpp, called using extern as theyre not defined here 
// init loads engine and allocates gpu buffers 
// infer runs one forward pass, retruns raw yolo output 
extern "C" cv::Mat trtInfer(const cv::Mat& blob); 
extern "C" void trtInit(const std::string& plan);
// simple detection structure for after post processing 
// box = bounding box in orginal image coords 
// cls = predicted class index (COCO)
// score = confidence score for each class 
// distance_m now added to be filled after the rest from lidar data 

// now added - pose models output keypoints 
struct Det {
    Rect box;
    int cls;
    float score;
    float distance_m = -1.0f;

    // Pose keypoints
    std::vector<cv::Point2f> kpts;   // size 17 - keypoints XY
    std::vector<float>       ksc;    // size 17 - conf of keypoitns 
};

// COCO17 skeleton pairs   
// these points describe how everythin connects for drawing stage
static const int COCO17_PAIRS[][2] = {
    {5,7},{7,9},      // left arm
    {6,8},{8,10},     // right arm
    {11,13},{13,15},  // left leg
    {12,14},{14,16},  // right leg
    {5,6},{5,11},{6,12},{11,12}, // torso
    {0,1},{0,2},{1,3},{2,4},      // face
    {0,5},{0,6}       // head to shoulders
};
static const int COCO17_NUM_PAIRS = sizeof(COCO17_PAIRS) / sizeof(COCO17_PAIRS[0]);

bool use_lidar = true;

// Convert bbox centre x-coordinate into a horizontal angle in radians,
// uses the camera horizontal FOV in degrees.
// full camera FOV is 62 degrees as per raspberry pi docs 
float bbox_center_angle(const cv::Rect& box, int img_width, float cam_fov_deg)
{
    // bbox centre x in pixels
    // this will be 0 degrees 
    float x_c = box.x + 0.5f * box.width;

    // normalised x in -1, 1, where 0 = image centre
    // left edge nx = -1 right edge nx = +1
    float nx = (x_c - img_width / 2.0f) / (img_width / 2.0f);

    // map to angle in radians using camera FOV
    const float PI_F = 3.1416f;
    float half_fov_rad = (cam_fov_deg * PI_F / 180.0f) * 0.5f; // this gives angle from the centre of frame

    //   multiply it by + or - depending if its off the centre to left or right then   
    return nx * half_fov_rad;
}


// Convert an x pixel coordinate into a horizontal angle in radians
static inline float x_pixel_to_angle(float x_px, int img_width, float cam_fov_deg)
{
    float nx = (x_px - img_width / 2.0f) / (img_width / 2.0f);

    const float PI_F = 3.1416f;
    float half_fov_rad = (cam_fov_deg * PI_F / 180.0f) * 0.5f; // this gives angle from the centre of frame

    return nx * half_fov_rad;
}


// from a lidar scan of loads of angles and distnace points , we need to give best distance near the target angle from cam angle of bbox
// scan is a vector of all lidar point from this scan, as  (angle_rad, distance_m)
float distance_from_scan_for_angle(const std::vector<LidarPoint>& scan,
                                   float target_angle_rad, // target angle is the angle we want to find a dist at
                                   float angle_window_deg, // this is how wide around the angle that we can search
                                   float max_range_m = 12.0f) //ignore anything further than this, 12m is max according to datasheet
{
    const float PI_F = 3.1416f;
    float window = angle_window_deg * PI_F / 180.0f; // just convert angle window into radians 

    float best = -1.0f; // initialise this to a negative number, to say havent foun anything yet 
    // loop through every point in scan
    for (const auto& p : scan) { 
        float a = p.first; // angle of point 
        float d = p.second; // dist of point
        if (d <= 0.0f || d > max_range_m) continue; // just chcek that the distnace is between 0 and 12 m
        if (std::fabs(a - target_angle_rad) <= window) { // if angle is within target angle 
            if (best < 0.0f || d < best) best = d; //update best with the closest distance in the angle window
        }
    }
    return best;
}

// helper for robust lidar dist,  collect all distances within an angular span, trim extremes thrn take median
float distance_from_scan_for_angle_span_trimmed_median(const std::vector<LidarPoint>& scan,
                                                       float a0_rad,
                                                       float a1_rad,
                                                       float trim_frac = 0.2f,
                                                       float max_range_m = 12.0f,
                                                       int min_pts = 5)
{
    //  check if angle a is inside [lo, hi] when range might wrap around +-pi
    auto in_span = [](float a, float lo, float hi) -> bool {
        if (lo <= hi) return (a >= lo && a <= hi);
      
        return (a >= lo || a <= hi);
    };

    std::vector<float> ds;
    ds.reserve(scan.size());

    for (const auto& p : scan) {
        float a = p.first;  // angle of point
        float d = p.second; // dist of point
        if (d <= 0.0f || d > max_range_m) continue;
        if (in_span(a, a0_rad, a1_rad)) ds.push_back(d);
    }

    if ((int)ds.size() < min_pts) return -1.0f;

    std::sort(ds.begin(), ds.end());

    int n = (int)ds.size();
    int trim = (int)std::floor(n * trim_frac);
    int lo = trim;
    int hi = n - trim; 

    // ensure we still have something left after trimming
    if (hi - lo < 1) { lo = 0; hi = n; }

    int m = hi - lo;
    int mid = lo + m / 2;

    // median of trimmed slice
    if (m % 2 == 1) {
        return ds[mid];
    } else {
        return 0.5f * (ds[mid - 1] + ds[mid]);
    }
}


// sigmoid helper for models if they output logits , instead of probs 
static inline float sigmoid(float x){ return 1.f/(1.f+expf(-x)); }

// Resize image to fit into new_shape while preserving aspect ratio
// then pad with colour so the result is exactly new_shape.
// r = scale factor from original to resized
// top = vertical padding
//left= horizontal padding
// These are later used to map detections back to original image size.
void letterbox(const Mat& img, Mat& out, Size new_shape, Scalar colour, float& r, int& top, int& left){
    // computes the scale factor for the image without distorting the image
    // takes the smaller of the scale needed to fit height or the scale needed to fit with 
    float r0 = min(new_shape.width/(float)img.cols, new_shape.height/(float)img.rows);
    int nw = round(img.cols*r0), nh = round(img.rows*r0); // computes new width and height after scalign
    Mat resized; resize(img, resized, Size(nw,nh)); // resize the image 
    out = Mat(new_shape, img.type(), colour); // creates new blank image , filled w colour padding
    top=(new_shape.height-nh)/2; 
    left=(new_shape.width-nw)/2; // how mich padding is required on left in pixels 
    resized.copyTo(out(Rect(left,top,nw,nh))); // now copy resized image into middle of padded output image 
    r = r0; // retrun the scale factor for further on when using yolo boxes 
}

// simple helper to go from degrees to rads
static float deg2radf(float deg) {
	 return deg * 3.14159265f / 180.0f;
	 }

     // this helper wraps an angle in rads to +-pi , to be used around the centre point 0 of lidar
static float wrap_rad_pm_pi(float a) {
    const float PI = 3.1416f;
    while (a >  PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

// these helpers will allow for a symbol be placed on pedestrian to indicate if facing cam or away
// uses the points on face, if they are in view symbol will will be a nought, if they arent it will be a cross
// this first helper checks that the points on the body are all in order 
static inline bool kpt_ok(const Det& d, int idx, float th=0.30f) {
    return idx >= 0 && idx < (int)d.kpts.size() && d.ksc[idx] >= th;
}

static void draw_torso_normal_symbol(cv::Mat& img, const Det& d) {
    const float th = 0.30f;
    const int LS=5, RS=6, LH=11, RH=12, NOSE=0, LE=1, RE=2;

    if (!(kpt_ok(d,LS,th) && kpt_ok(d,RS,th) && kpt_ok(d,LH,th) && kpt_ok(d,RH,th))) return;

    cv::Point2f S = 0.5f * (d.kpts[LS] + d.kpts[RS]); // shoulder centres 

    // boolean value for if the face is visible or not
    // confidence thgresholds of each can be finetuned 
    bool face_visible = kpt_ok(d, NOSE, 0.35f) || (kpt_ok(d,LE,0.35f) && kpt_ok(d,RE,0.35f));

    // fallback is the  confidence imbalance between body sides
    float right_conf = (d.ksc[RS] + d.ksc[RH]);
    float left_conf  = (d.ksc[LS] + d.ksc[LH]);
    bool likely_front = face_visible; 
    if (!face_visible) {
        // If one side is consistently lower conf, person is likely turned so we cant know sign reliably
        // Treat uncertain as facing screen to avoid overconfidence
        likely_front = false;
    }

    // draw symbol,  circle + dot  or circle + X 
    int R = std::max(10, (int)(0.08f * d.box.width));
    cv::circle(img, S, R, cv::Scalar(0,255,255), 2, cv::LINE_AA);

    if (likely_front) {
        cv::circle(img, S, std::max(2, R/4), cv::Scalar(0,255,255), -1, cv::LINE_AA); // dot
    } else {
        cv::line(img, S + cv::Point2f(-R*0.6f, -R*0.6f), S + cv::Point2f(R*0.6f, R*0.6f),
                 cv::Scalar(0,255,255), 2, cv::LINE_AA);
        cv::line(img, S + cv::Point2f(-R*0.6f, R*0.6f), S + cv::Point2f(R*0.6f, -R*0.6f),
                 cv::Scalar(0,255,255), 2, cv::LINE_AA);
    }
}



// helper to convert ByteTrack recangle to  cv::Rect 
static inline cv::Rect btrect_to_cv(const byte_track::Rect<float>& r, int W, int H) {
    int x = (int)std::round(r.x());
    int y = (int)std::round(r.y());
    int w = (int)std::round(r.width());
    int h = (int)std::round(r.height());
    cv::Rect box(x, y, w, h);
    return box & cv::Rect(0, 0, W, H);
}

static inline float iou_rect(const cv::Rect& a, const cv::Rect& b) {
    float inter = (float)(a & b).area();
    float uni = (float)(a.area() + b.area() - inter) + 1e-6f;
    return inter / uni;
}

// match a track bbox to the best detection bbox this frame (to reuse pose and lidar distance)
static int match_det_by_iou(const cv::Rect& tbox, const std::vector<Det>& dets,
                            float min_iou = 0.3f) {
    int best = -1;
    float best_i = min_iou;
    for (int i = 0; i < (int)dets.size(); i++) {
        if (dets[i].cls != 0) continue; 
        float iou = iou_rect(tbox, dets[i].box);
        if (iou > best_i) { best_i = iou; best = i; }
    }
    return best;
}





// Entry point
//  open camera , initialise TRT engine,
//  process frames in a loop

// debug statements can be ignored, was just added trying to find where a segmentation issue was when starting up
int main(int argc, char** argv){
std::cout << "[DEBUG] main() start" << std::endl;

// GStreamer pipeline used by opencv video capturing
 // If a pipeline string is passed as argv[1], use that, otherwise use default CSI camera pipeline
// done this incase pipeline is chnaged later in proj 

	cv::VideoCapture cap;

	if (argc > 1) {
	    std::string src = argv[1];

	    if (src.find("!") != std::string::npos) {
	        // GStreamer pipeline (from run.sh)
	        cap.open(src, cv::CAP_GSTREAMER);
	    } else {
	        // Video file path -> use GStreamer with a proper URI
	        std::string uri = "file://" + src;
		std::string gst =
   		 "filesrc location=" + src +
		    " ! qtdemux ! h264parse ! avdec_h264 "
		    " ! videoconvert ! video/x-raw,format=BGR "
		    " ! appsink drop=1 max-buffers=1";
		cap.open(gst, cv::CAP_GSTREAMER);
	    }
	} else {
	    std::string default_pipe =
	        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1,format=NV12 "
	        "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR "
	        "! appsink drop=1 max-buffers=1";
	    cap.open(default_pipe, cv::CAP_GSTREAMER);
	}

	if(!cap.isOpened()){
	    std::cerr << "Failed to open video source\n";
	    return -1;
	}
	
	

	// --no display flag , dont open view on screen , increases fps slightly 
	bool show = true;
	bool lidar_debug = false;
	bool bypass_letterbox = false;
	bool disable_lidar = false; 
        for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--no-display") show = false;
	if (string(argv[i]) == "--lidar-debug") lidar_debug = true;
	if (string(argv[i]) == "--bypass-letterbox") bypass_letterbox = true;
	if (string(argv[i]) == "--no-lidar") disable_lidar = true;

    }


 std::cout << "[DEBUG] camera opened OK" << std::endl;
// inp use 416 network input size 
//conf = confidence thresh to keep a det
//nms_th = IoU thresh for nms 
    const int INP=416; 
    const float conf_th=0.15f, 
    nms_th=0.5f;
// load trt engine , done once and reused all frames
  std::cout << "[DEBUG] before trtInit" << std::endl;
    trtInit("/home/sean/models/yolov8n/yolov8n_pose_416_fp16.plan"); // init using the .plan file path
    
    // ByteTrack tracker from pulled repo 
byte_track::BYTETracker tracker(
    30,   // frame_rate
    60,   // track_buffer
    0.15f, // track_thresh
    0.25f, // high_thresh
    0.8f  // match_thresh
);

std::cout << "[DEBUG] after trtInit (before lidar)" << std::endl;

// optional lidar (only constructed if enabled)
std::unique_ptr<LidarC1> lidar;

if (use_lidar && !disable_lidar) {
    std::cout << "[DEBUG] initialising lidar" << std::endl;
    lidar = std::make_unique<LidarC1>("/dev/ttyUSB0", 460800);
    std::cout << "[DEBUG] after lidar init" << std::endl;
} else {
    std::cout << "[DEBUG] lidar disabled" << std::endl;
}
   

// main procesing loop - grab frame , run inference , postprocess, draw and log fps
    Mat frame; //opencv container called frame, each loop cap.read(frame) will fill this 
    int frames=0; // counter for how many frame sprocessed 
    int64 last=getTickCount(); //timer for process used ltaer on 
    
   std::cout << "[DEBUG] entering frame loop" << std::endl;
    while(true){
// read one fram from camera , break loop if stream ends or frame invalid
        if(!cap.read(frame) || frame.empty()) break; // if frames empty or not read right break from loop 

	// letterbox as defined above to 416x416
        float ratio = 1.0f; 
        int top = 0, left = 0; 
        Mat inp = frame;

        if (!bypass_letterbox) {
            Mat lb; 
            letterbox(frame, lb, Size(INP,INP), Scalar(114,114,114), ratio, top, left);
            inp = lb;
        } else {
            // bypass letterbox: assumes pipeline outputs INP×INP already (GPU resize distort path)
            ratio = 1.0f; top = 0; left = 0;
            inp = frame;
        }

        // this converts to a blob tensor used by yolo 
        Mat blob = dnn::blobFromImage(inp, 1/255.0, Size(INP,INP), Scalar(), true, false); // NCHW float32
        Mat out = trtInfer(blob); // run inference on the blob and retrun a matrix shaped 84xN


	// prepare for decoding the output
 	// pose only looks for humans so no need for coco80 classes anymore
	// it now has 56 channels , 17 points x 3 = 51 , + 4 coords for box , + 1 overall conf
	const int N = out.cols;          // number of candidates
	const int C = out.rows;          // channels per candidate - 84 for detect, 56 for pose
	const float* p = (const float*)out.data;

	static bool printed_shape = false;
	if (!printed_shape) {
	    std::cout << "[DEBUG] TRT out shape: rows=" << C << " cols=" << N << std::endl;
	    printed_shape = true;
	}

	vector<Det> dets;
	dets.reserve(N);

	for (int i = 0; i < N; i++) {
	    //the box coords  are always the first 4 channels
	    float cx = p[0 * N + i];
	    float cy = p[1 * N + i];
	    float w  = p[2 * N + i];
	    float h  = p[3 * N + i];
		
		// pose here is only human, so class id is fixed at 0
		// confidence read from channel 4
	    int   best  = 0;
	    float bestp = 0.f;
		// left in here the previous decode for yolov8n model in case of swithcing back 
	    if (C == 84) {
	        // YOLOv8 detect  4 + 80 class scores
	        best = -1;
	        bestp = 0.f;
	        for (int c = 0; c < 80; c++) {
	            float s = p[(4 + c) * N + i];
	            if (s > bestp) { bestp = s; best = c; }
	        }
	        if (bestp < conf_th) continue;
	    } else {
	        // pose = 56
	        best  = 0;                 // person
	        bestp = p[4 * N + i];      // person confidence
	        if (bestp < conf_th) continue;
	    }

	    // convert from center to top-left in model coords
	    float x = cx - w * 0.5f;
	    float y = cy - h * 0.5f;

	    float rx = x, ry = y, rw = w, rh = h;

	    // if letterbox was used, undo padding+scale to map back to original frame
	    if (!bypass_letterbox) {
	        rx = (x - left) / ratio;
	        ry = (y - top)  / ratio;
	        rw = w / ratio;
	        rh = h / ratio;
	    }
	    Rect box = Rect(cvRound(rx), cvRound(ry), cvRound(rw), cvRound(rh)) &
	           Rect(0, 0, frame.cols, frame.rows);

	   if (box.area() > 0) {
	    Det d;
	    d.box = box;
	    d.cls = best;
	    d.score = bestp;

	    // If pose output, extract keypoints and store them 
	    // 	keypoints are stored as x, y , score... starting at channel 5
            // also has to under padding or letterbox for keypoints same as bboxes
	    if (C >= 56 && best == 0) {
	        const int kpt_start = 5;
	        const int kpt_step  = 3;
	        const int num_kpts  = (C - kpt_start) / kpt_step;  // expect 17
	
	        d.kpts.assign(num_kpts, cv::Point2f(-1, -1));
	        d.ksc.assign(num_kpts, 0.0f);

	        for (int k = 0; k < num_kpts; k++) {
	            float kx = p[(kpt_start + k * kpt_step + 0) * N + i];
	            float ky = p[(kpt_start + k * kpt_step + 1) * N + i];
	            float ks = p[(kpt_start + k * kpt_step + 2) * N + i];

	            // map back to original frame if letterbox used
	            float fx = kx, fy = ky;
	            if (!bypass_letterbox) {
	                fx = (kx - left) / ratio;
	                fy = (ky - top)  / ratio;
	            }

	            d.kpts[k] = cv::Point2f(fx, fy);
	            d.ksc[k]  = ks;
	        }
	    }
	
	    dets.push_back(std::move(d));
	}
	
}

	// NMS
    // first sort the candidates by highest conf 
	sort(dets.begin(), dets.end(),
	     [](const Det& a, const Det& b){ 
            return a.score > b.score; });

	vector<int> keep; // index of boxes to keep 
	vector<char> rem(dets.size(), 0); // rem[i] = 1 means suppress the box 
	
    // greedy keep and suppress 
    // go through boxes in score order, if box i isnt removed keep it 
	for (size_t i = 0; i < dets.size(); ++i) {
	    if (rem[i]) continue;
	    keep.push_back((int)i);
        
        // compare kept box to all lower scores 
        // for each later box j of lower score check its overlap with box i
	    for (size_t j = i + 1; j < dets.size(); ++j) {
	        if (rem[j]) continue;
            // box & box can find the intersection triangle then , held by inter 
	        float inter = (dets[i].box & dets[j].box).area();
	        float iou = inter / (dets[i].box.area() + dets[j].box.area() - inter + 1e-6f); // iou is the ratio of overlap 
	        if (iou > nms_th) rem[j] = 1; // then suppress if the overlap is too high
	    }
	}
	

	// DEBUG stage ---  draw raw detections beofre bytetrack to see if YOLO is actually detecting them
for (int id : keep) {
    const Det& d = dets[id];
    if (d.cls != 0) continue;
    cv::rectangle(frame, d.box, cv::Scalar(0,0,255), 2); // red raw dets
    putText(frame, cv::format("DET %.2f", d.score), d.box.tl() + cv::Point(0,-3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
}


	
	// update bytetraxck
	std::vector<byte_track::Object> objects;
	objects.reserve(keep.size());

	for (int id : keep) {
	    const Det& d = dets[id];
	    if (d.cls != 0) continue; // track only people 
	    byte_track::Rect<float> r((float)d.box.x, (float)d.box.y,
	                             (float)d.box.width, (float)d.box.height);
	    objects.emplace_back(r, d.cls, d.score);
	}

	std::vector<byte_track::BYTETracker::STrackPtr> tracks = tracker.update(objects);

	const float cam_fov_deg = 62.0f; // cam angle from rpi docs 
        const float angle_window_deg = 3.0f; // this is the window around the centre of bbox we can search for points in 

        const float lidar_sign = +1.0f;         
        const float lidar_yaw_offset_deg = 0.0f; // this can be tuned once i have a stable rig, not temp one we have right now 
	
	static std::vector<LidarPoint> scan_cache;
	static int64 last_scan_tick = 0;                 // tick count of last scan update
	double scan_period_s = 0.10;               

	// grab a lidar scan for this frame
	if(use_lidar && lidar){	
	    //  only update the scan occasionally, reuse cached scan otherwise
	    int64 now_tick = cv::getTickCount();
	    double now_s = now_tick / cv::getTickFrequency();
	    double last_s = (last_scan_tick == 0) ? 0.0 : (last_scan_tick / cv::getTickFrequency());

	    if (last_scan_tick == 0 || (now_s - last_s) >= scan_period_s) {
	        int64 t0 = cv::getTickCount();
	        scan_cache = lidar->getScan();
	        double tscan = (cv::getTickCount() - t0) / cv::getTickFrequency();
		if(lidar_debug){
	        std::cout << "[TIMING] lidar.getScan() = " << (tscan * 1000.0) << " ms\n";
		}
	        last_scan_tick = now_tick;
	    }



	    // now assign a lidar distance to each kept detection
	    for (int id : keep) { // loop over indices that survived nms
	        Det& d = dets[id]; //get a refernece to that detection
	        if (d.cls != 0) continue; // only fuse for PERSON class 0, for now anyway Can be adjusted later on

		// convert bbox left/right edges into camera angles
		// use only middle band of bbox width so we dont pick up background either side of person
		// TOdo for this to work i need to make sure lidar and camera are perfectly aligned 
		float band_frac = 0.10f; // middle 10% - must be 10 now incase of just catching someones head
		float cx = d.box.x + 0.5f * d.box.width;
		float half = 0.5f * band_frac * d.box.width;
		float x0 = cx - half;
		float x1 = cx + half;

		// clamp to image bounds
		if (x0 < 0.0f) x0 = 0.0f;
		if (x1 > (float)(frame.cols - 1)) x1 = (float)(frame.cols - 1);

		float cam_a0 = x_pixel_to_angle(x0, frame.cols, cam_fov_deg);
		float cam_a1 = x_pixel_to_angle(x1, frame.cols, cam_fov_deg);
	        // convert cam angles into lidar angles
	        float lidar_a0 = wrap_rad_pm_pi(lidar_sign * cam_a0 + deg2radf(lidar_yaw_offset_deg));
	        float lidar_a1 = wrap_rad_pm_pi(lidar_sign * cam_a1 + deg2radf(lidar_yaw_offset_deg));

	        // small margin to tolerate slight misalignment 
		// this can more than likely be removed when i create a stable rig 
	        float margin = deg2radf(2.0f);
	        lidar_a0 = wrap_rad_pm_pi(lidar_a0 - margin);
	        lidar_a1 = wrap_rad_pm_pi(lidar_a1 + margin);
	
	        //  trimmed median over bbox angular span
	        float dist = distance_from_scan_for_angle_span_trimmed_median(scan_cache, lidar_a0, lidar_a1, 0.2f, 10.0f, 5);

	        // fallback if not enough points , use old centre window method
	        if (dist < 0.0f) {
	            float cam_angle = bbox_center_angle(d.box, frame.cols, cam_fov_deg); // convert bbox centre into camera angle in rads

	            // convert cam angle into lidar angle
	            // lidar sign handle left and right mismatch between camera and lidar coords
	            // yaw offset hasnt been dealt with as of yet
	            float lidar_angle = wrap_rad_pm_pi(lidar_sign * cam_angle + deg2radf(lidar_yaw_offset_deg));
	            // look into lidar scan for points near lidar angle within spec window
	            //retruns one closest
	            dist = distance_from_scan_for_angle(scan_cache, lidar_angle, angle_window_deg, 10.0f);
	        }


	            d.distance_m = dist;
	    } 
	} else { 
	    
	    for (int id : keep) dets[id].distance_m = -1.0f;
	}
	
	// Draw
    int persons = 0; // count of persons in frame 
    static int pf = 0;

    for (const auto& tp : tracks) { // loop over tracked persons
        if (!tp || !tp->isActivated()) continue;
        if (tp->getSTrackState() != byte_track::STrackState::Tracked) continue;

        cv::Rect tbox = btrect_to_cv(tp->getRect(), frame.cols, frame.rows);
        if (tbox.area() <= 0) continue;

        persons++;

        // Attach this track to a Det this frame (reuse pose + lidar distance)
        int di = match_det_by_iou(tbox, dets, 0.3f);
        if (di < 0) continue;

        Det& d = dets[di];
        if (d.cls != 0) continue; // onlt apply to people 

        // bbox centre into angle 
        float cam_angle = bbox_center_angle(d.box, frame.cols, cam_fov_deg);

        cv::rectangle(frame, tbox, cv::Scalar(0,255,0), 2); // draw bounding box in green 

        // draw track id
        putText(frame, cv::format("ID:%zu", tp->getTrackId()), tbox.tl() + cv::Point(0,-20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);

        // drawing on keypoints here 
        // connects the lines using the coco17_pairs from above 
        if (C >= 56 && !d.kpts.empty()) {
            const float kpt_th = 0.30f;

            // draw keypoints
            for (int k = 0; k < (int)d.kpts.size(); k++) {
                if (d.ksc[k] >= kpt_th) {
                    cv::circle(frame, d.kpts[k], 3, cv::Scalar(0, 0, 255), -1);
                }
            }

            // draw skeleton lines
            for (int pi = 0; pi < COCO17_NUM_PAIRS; pi++) {
                int a = COCO17_PAIRS[pi][0];
                int b = COCO17_PAIRS[pi][1];
                if (a < 0 ||  b < 0 ||  a >= (int)d.kpts.size() || b >= (int)d.kpts.size()) continue;

                if (d.ksc[a] >= kpt_th && d.ksc[b] >= kpt_th) {
                    cv::line(frame, d.kpts[a], d.kpts[b], cv::Scalar(255, 0, 0), 2);
                }
            }
        }
        if (!d.kpts.empty()) {   // here we can now draw on the symbol to indicate if they are facing away or not 
            draw_torso_normal_symbol(frame, d);
        }

        if ((++pf % 30) == 0) {
            std::cout << "[FUSE] det_angle_deg=" << (cam_angle * 180.0f / 3.14159265f)
                      << " dist=" << d.distance_m << "m\n";
        }

        // adding the distance measured onto the frame for visual display 
        std::string label;
        if (d.distance_m > 0.0f) // if there s distnace present display it 
            label = cv::format("ID:%zu %.2f %.1fm", tp->getTrackId(), d.score, d.distance_m);
        else
            label = cv::format("ID:%zu %.2f", tp->getTrackId(), d.score);

        //put all on frame 
        putText(frame, label, tbox.tl() + cv::Point(0,-3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }


        //FPS logging every second 
        frames++; 
        double dt=(getTickCount()-last)/getTickFrequency();
        if(dt>=1.0){ double fps=frames/dt;
            putText(frame, format("FPS: %.1f, persons: %d", fps, persons), Point(10,25),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
            cout<<"FPS: "<<fps<<" persons/frame: "<<persons<<endl;
            last=getTickCount(); frames=0;
        }
	// display annotated frame in a window 
        if(show){
	    cv::Mat disp;
            cv::resize(frame, disp, cv::Size(), 0.5, 0.5);  // 50% scale
            imshow("avdet", disp);
            if(waitKey(1)==27) break;
        }
        
    }
    return 0;
}

