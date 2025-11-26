// main application -- captures framaes from CSI cam , runs YOLOv8 TRT engine 
// decode detections , runs NMS , draw boxes , logs FPS and persons count 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <iostream>
using namespace cv; using namespace std;

// TRT wrappers from trt_yolo.cpp
// init loads engine and allocates gpu buffers 
// infer runs one forward pass, retruns raw yolo output 

extern "C" cv::Mat trtInfer(const cv::Mat& blob);
extern "C" void trtInit(const std::string& plan);
// simple detection structure for after post processing 
// box = bounding box in orginal image coords 
// cls = predicted class index (COCO)
// score = confidence score for each class 
struct Det { Rect box; int cls; float score; };

// sigmoid helper for models if they output logits , instead of probs 
static inline float sigmoid(float x){ return 1.f/(1.f+expf(-x)); }

// Resize image to fit into new_shape while preserving aspect ratio
// then pad with colour so the result is exactly new_shape.
// r = scale factor from original to resized
// top = vertical padding
//left= horizontal padding
// These are later used to map detections back to original image size.
void letterbox(const Mat& img, Mat& out, Size new_shape, Scalar color, float& r, int& top, int& left){
    float r0 = min(new_shape.width/(float)img.cols, new_shape.height/(float)img.rows);
    int nw = round(img.cols*r0), nh = round(img.rows*r0);
    Mat resized; resize(img, resized, Size(nw,nh));
    out = Mat(new_shape, img.type(), color);
    top=(new_shape.height-nh)/2; left=(new_shape.width-nw)/2;
    resized.copyTo(out(Rect(left,top,nw,nh)));
    r = r0;
}
// Entry point
//  open camera , initialise TRT engine,
//  process frames in a loop
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
    if(!cap.isOpened()){ cerr<<"Failed to open CSI camera via GStreamer\n"; return -1; }
 std::cout << "[DEBUG] camera opened OK" << std::endl;
// inp use 416 network input size 
//conf = confidence thresh to keep a det
//nms_th = IoU thresh for nms 
    const int INP=416; const float conf_th=0.25f, nms_th=0.5f;
// load trt engine , done once and reused all frames
  std::cout << "[DEBUG] before trtInit" << std::endl;
    trtInit("/home/sean/models/yolov8n/yolov8n_fp16.plan");
    std::cout << "[DEBUG] after trtInit" << std::endl;

    


// main procesing loop - grab frame , run inference , postprocess, draw and log fps
    Mat frame; int frames=0; int64 last=getTickCount();
    
   std::cout << "[DEBUG] entering frame loop" << std::endl;
    while(true){
// read one fram from camera , break loop if stream ends or frame invalid
        if(!cap.read(frame) || frame.empty()) break;

        float ratio; int top,left; Mat lb; letterbox(frame, lb, Size(INP,INP), Scalar(114,114,114), ratio, top, left);
        Mat blob = dnn::blobFromImage(lb, 1/255.0, Size(INP,INP), Scalar(), true, false); // NCHW float32

        Mat out = trtInfer(blob); // [84,N]
        const int N = out.cols; const float* p = (float*)out.data;
        vector<Det> dets; dets.reserve(N);
        for(int i=0;i<N;i++){
            float cx=p[0*N+i], cy=p[1*N+i], w=p[2*N+i], h=p[3*N+i];
            int best=-1; float bestp=0.f;
            for(int c=0;c<80;c++){ float s=p[(4+c)*N+i]; if(s<0||s>1) s=sigmoid(s); if(s>bestp){bestp=s; best=c;} }
            if(bestp<conf_th) continue;
            float x=cx-w*0.5f, y=cy-h*0.5f;
            float rx=(x-left)/ratio, ry=(y-top)/ratio, rw=w/ratio, rh=h/ratio;
            Rect box = Rect(cvRound(rx),cvRound(ry),cvRound(rw),cvRound(rh)) & Rect(0,0,frame.cols,frame.rows);
            if(box.area()>0) dets.push_back({box,best,bestp});
        }
        // NMS
	// sort detections by descending order
	// greedily keep highest score boxes remove any with iou less than nms_thresh
        sort(dets.begin(), dets.end(), [](const Det&a,const Det&b){return a.score>b.score;});
        vector<int> keep; vector<char> rem(dets.size(),0);
        for(size_t i=0;i<dets.size();++i){ if(rem[i]) continue; keep.push_back((int)i);
            for(size_t j=i+1;j<dets.size();++j){ if(rem[j]) continue;
                float inter=(dets[i].box & dets[j].box).area();
                float iou = inter / (dets[i].box.area()+dets[j].box.area()-inter+1e-6f);
                if(iou>nms_th) rem[j]=1; } }
	// count persons coco class 0
        int persons=0;
        for(int id: keep){
            auto& d=dets[id]; if(d.cls==0) persons++;
            rectangle(frame, d.box, Scalar(0,255,0), 2);
            putText(frame, to_string(d.cls)+format(" %.2f",d.score), d.box.tl()+Point(0,-3),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
        }
        // FPS logging every second 
        frames++; double dt=(getTickCount()-last)/getTickFrequency();
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
