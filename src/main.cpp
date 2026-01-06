// main application -- captures framaes from cam , runs YOLOv8 TRT engine 
// decode detections , runs NMS , draw boxes , logs FPS and persons count 

// added now: lidar fusion - computes the angle of where box is in cam fov and maps that to lidar angle and tags it with its distance 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include "lidar_c1.hpp"
#include <iostream>
#include <cmath>
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
struct Det { Rect box; int cls; float score; float distance_m = -1.0f; };



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



// Entry point
//  open camera , initialise TRT engine,
//  process frames in a loop

// debug statements can be ignored, was just added trying to find where a segmentation issue was when starting up
int main(int argc, char** argv){
std::cout << "[DEBUG] main() start" << std::endl;

// GStreamer pipeline used by opencv video capturing
 // If a pipeline string is passed as argv[1], use that, otherwise use default CSI camera pipeline
// done this incase pipeline is chnaged later in proj 
    string pipe = (argc>1)? argv[1] :
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1,format=NV12 "
        "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR "
        "! appsink drop=1 max-buffers=1";

// --no display flag , dont open view on screen , increases fps slightly 
	bool show = true;
        for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--no-display") show = false;
    }


    VideoCapture cap(pipe, CAP_GSTREAMER);
    if(!cap.isOpened()){ cerr<<"Failed to open CSI camera via GStreamer\n"; return -1; } // just print an error in case it doesnt open right
 std::cout << "[DEBUG] camera opened OK" << std::endl;
// inp use 416 network input size 
//conf = confidence thresh to keep a det
//nms_th = IoU thresh for nms 
    const int INP=416; 
    const float conf_th=0.25f, 
    nms_th=0.5f;
// load trt engine , done once and reused all frames
  std::cout << "[DEBUG] before trtInit" << std::endl;
    trtInit("/home/sean/models/yolov8n/yolov8n_fp16.plan"); // init using the .plan file path 
    LidarC1 lidar("/dev/ttyUSB0", 460800);     // start the lidar 
    std::cout << "[DEBUG] after trtInit" << std::endl;

    


// main procesing loop - grab frame , run inference , postprocess, draw and log fps
    Mat frame; //opencv container called frame, each loop cap.read(frame) will fill this 
    int frames=0; // counter for how many frame sprocessed 
    int64 last=getTickCount(); //timer for process used ltaer on 
    
   std::cout << "[DEBUG] entering frame loop" << std::endl;
    while(true){
// read one fram from camera , break loop if stream ends or frame invalid
        if(!cap.read(frame) || frame.empty()) break; // if frames empty or not read right break from loop 

        // letterbox as defined above to 416x416
        float ratio; 
        int top,left; 
        Mat lb; 
        letterbox(frame, lb, Size(INP,INP), Scalar(114,114,114), ratio, top, left);

        // this converts to a blob tensor used by yolo 
        Mat blob = dnn::blobFromImage(lb, 1/255.0, Size(INP,INP), Scalar(), true, false); // NCHW float32

        Mat out = trtInfer(blob); // run inference on the blob and retrun a matrix shaped 84xN

        // prepare for decoding the output 
        const int N = out.cols;   // N is the number of candidate predictions 
        const float* p = (float*)out.data; //p points to the raw float data inside out
        vector<Det> dets; // dets will store decoded detections
        dets.reserve(N); // this avoids reallocations 

        for(int i=0;i<N;i++){
            // for every candidate get the box coordinates centre x and y and wdith and height 
            float cx=p[0*N+i], cy=p[1*N+i], w=p[2*N+i], h=p[3*N+i];
            int best=-1; // best class id
            float bestp=0.f; // best class confidence 
            // this loop then assigns the best suited label from 80 COCO 
            for(int c=0;c<80;c++){
                 float s=p[(4+c)*N+i]; 
                 //if(s<0||s>1) s=sigmoid(s); 
                 if(s>bestp){bestp=s; best=c;} 
                }
            if(bestp<conf_th) continue; // drop all the weak preditctions 
            float x=cx-w*0.5f, y=cy-h*0.5f; // this converts from centre to top left coords 
            float rx=(x-left)/ratio, ry=(y-top)/ratio, rw=w/ratio, rh=h/ratio; // map from letterboxed back to originals, subtract padding, divide byratio
            Rect box = Rect(cvRound(rx),cvRound(ry),cvRound(rw),cvRound(rh)) & Rect(0,0,frame.cols,frame.rows);
            if(box.area()>0) dets.push_back({box,best,bestp}); // keep only valid boxes and store for later fusion 
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

	// grab a lidar scan for this frame 
	auto scan = lidar.getScan();

    // now assign a lidar distance to each kept detection
	const float cam_fov_deg = 62.0f; // cam angle from rpi docs 
	const float angle_window_deg = 3.0f; // this is the window around the centre of bbox we can search for points in 
	
	const float lidar_sign = +1.0f;         
	const float lidar_yaw_offset_deg = 0.0f; // this can be tuned once i have a stable rig, not temp one we have right now 

	for (int id : keep) { // loop over indices that survived nms 
	    Det& d = dets[id]; //get a refernece to that detection
	    if (d.cls != 0) continue; // only fuse for PERSON class 0, for now anyway Can be adjusted later on 

	    float cam_angle = bbox_center_angle(d.box, frame.cols, cam_fov_deg); // convert bbox centre into camera angle in rads 
	
        // convert cam angle into lidar angle 
        // lidar sign handle left and right mismatch between camera and lidar coords 
        // yaw offset hasnt been dealt with as of yet 
	    float lidar_angle = wrap_rad_pm_pi(lidar_sign * cam_angle + deg2radf(lidar_yaw_offset_deg));
        // look into lidar scan for points near lidar angle within spec window
        //retruns one closest 
	    d.distance_m = distance_from_scan_for_angle(scan, lidar_angle, angle_window_deg, 10.0f);
	}
	


	// Draw
	int persons = 0; // count of persons in frame 
	static int pf = 0;

	for (int id : keep) { // loop over kept detectiosn
	    Det& d = dets[id];
	    if (d.cls != 0) continue; // onlt apply to people 
	    persons++;
        // bbox centre into angle 
	    float cam_angle = bbox_center_angle(d.box, frame.cols, cam_fov_deg);

	    cv::rectangle(frame, d.box, cv::Scalar(0,255,0), 2); // draw bounding box in green 

	    if ((++pf % 30) == 0) {
	        std::cout << "[FUSE] det_angle_deg=" << (cam_angle * 180.0f / 3.14159265f)
	                  << " dist=" << d.distance_m << "m\n";
	    }

        // adding the distance measured onto the frame for visual display 
	    std::string label;
	    if (d.distance_m > 0.0f) // if there s distnace present display it 
	        label = cv::format("%d %.2f %.1fm", d.cls, d.score, d.distance_m);
	    else
	        label = cv::format("%d %.2f", d.cls, d.score);
        
            //put all on frame 
	    putText(frame, label, d.box.tl() + cv::Point(0,-3),
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
	}

        // FPS logging every second 
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
            imshow("avdet", frame);
            if(waitKey(1)==27) break;
        }
        
    }
    return 0;
}

