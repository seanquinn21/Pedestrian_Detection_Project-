// trt_yolo.cpp
// Minimal TensorRT wrapper:
//  - trtInit(planPath): loads the serialized TensorRT engine (.plan) and allocates GPU buffers
//  - trtInfer(blob): copies input blob to GPU, runs inference, copies output back, returns as cv::Mat

#include <NvInfer.h> // this is TRT API 
#include <cuda_runtime_api.h> // this where we include our cuda functions 

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

using namespace nvinfer1; // remove the need for repeating nvinfer when calling any classes/interfaces in it

// The logger must be implemented
// its an interface inside tensorRTs namespace nvinfer
// implmemented here as its defined 
// noexecpt is reuqired as per nvidia docs to ensure it matches the interface, override lets compiler ensure it matches
struct TRTLogger : public ILogger {
    void log(Severity s, const AsciiChar* msg) noexcept override {
        // this check ensures we are only printing warnings or errors, ignore enums in ILogger that arent as severe as warning
        if (s <= Severity::kWARNING) {
          // output the error , TRT appended to specify where the error is originating 
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static TRTLogger gLogger; // then we can create one global logger obj 

// want a custom clean up object 
// saves having a different deleter for IRunTime, ICUDAEngine, IExecutionContext 
// just releases all resources being used by an object 
struct TRTDestroy {
    template <typename T> // this means itll be called on any TRT type if its given a pointer to it 
    void operator()(T* t) const {  // operator() means it can be called like a function
        if (t) t->destroy();
    }
};

template <typename T> // template T - use any type given 
using TRTUnique = std::unique_ptr<T, TRTDestroy>; //TRTUnique is like a nickname here so ya can leave out 'std::unique_ptr<nvinfer1'... each time 

// this is a wrapper for the trt engine 
class TRTEngine {
public:
    // here takes in the file path to the .plan engine , the optimised plan for gpu 
    explicit TRTEngine(const std::string& planPath) {
        // reads the .plan file bytes into RAM as a vector<char>
        std::vector<char> planData = readFile(planPath);
        // create TRT runtime obj, runtime_ class member of TRTUnique<IRuntime> runtime_
        runtime_.reset(createInferRuntime(gLogger)); // createInferRuntime comes from nvinfer namespace 
        if (!runtime_){ // if it fails throw a runtime error here 
          throw std::runtime_error("createInferRuntime failed");
        }
        //takes the raw .plan bytes and reconstructs the compiled TensorRT engine object in memory
        engine_.reset(runtime_->deserializeCudaEngine(planData.data(), planData.size()));
        if (!engine_){ // if it cant be loaded throw and error 
          throw std::runtime_error("deserializeCudaEngine failed ");
        }
        // create execution context 
        // prepare the engine to be runnable for each frame 
        context_.reset(engine_->createExecutionContext()); // use pointer from engine_ 
        if (!context_) {
          throw std::runtime_error("createExecutionContext failed"); // throw error if needed 
        }

        // called from private method outlined below 

        //figures out which TensorRT binding index is input and which is output
        // binding is just the index that correspinds to a tensor , input or output 
        findBindings();

        // computing how many floats each binding has and allocate GPU buffers
        // allocates GPU memory for the input tensor and the output tensor 
        allocateBuffers();

        // Try to infer  number of predictions for YOLO style output
        inferOutN();
    }

    // Destructor
    //  runs when TRTEngine is destroyed
    // used to clean up GPU resources that were allocated 
    ~TRTEngine() {
        // Free GPU buffers
        if (dIn_)  cudaFree(dIn_); // if input buffer exists free it 
        if (dOut_) cudaFree(dOut_); // if output buffer exists free it 
    }

    // run one inference
    cv::Mat infer(const cv::Mat& blob) {
        // blob is the preprocessed input tensor in CPU memory 
        // (float32, NCHW)


        if (!context_) return cv::Mat(); // if theres no valid context, retrun empty output

        // Bindings array passed to TRT
        void* bindings[2] = {nullptr, nullptr};
        bindings[inBinding_]  = dIn_; //the input binding should read from GPU buffer dIn
        bindings[outBinding_] = dOut_; //the output binding should read from GPU buffer dOut

        // Copy input from CPU to GPU , using cpu pointer to blob data 
        cudaMemcpy(dIn_, blob.ptr<float>(), inBytes_, cudaMemcpyHostToDevice);

        // Execute inference , default cuda stream 0 
        // bindings is an array of pointers for all inputs and outputs 
        context_->enqueueV2(bindings, 0, nullptr);

        // Copy output from GPU back again to  CPU
        // top line here is allocating sufficienlty sized array to hold the tensor 
        std::vector<float> host(outBytes_ / sizeof(float));
        cudaMemcpy(host.data(), dOut_, outBytes_, cudaMemcpyDeviceToHost);


       // Return output as cv::Mat(C, N) regardless of whether engine gives [C,N] or [N,C]
	if (outC_ <= 0 || outN_ <= 0) {
	    throw std::runtime_error("inferOutN() did not set outC_/outN_");
	}

	if (host.size() != static_cast<size_t>(outC_ * outN_)) {
	    std::cerr << "[TRT] Unexpected output size. host floats=" << host.size()
	              << " expected=" << (outC_ * outN_) << " (C=" << outC_ << " N=" << outN_ << ")\n";
	    return cv::Mat();
	}

	if (outIsCN_) {
	    // [C, N]
	    cv::Mat m(outC_, outN_, CV_32F, host.data());
	    return m.clone();
	} else {
	    // [N, C] -> transpose to [C, N]
	    cv::Mat raw(outN_, outC_, CV_32F, host.data());
	    cv::Mat t;
	    cv::transpose(raw, t); // now (C, N)
	    return t.clone();
	}
       


    }

private:
    // 3 core objects for this wrapper 
    TRTUnique<IRuntime> runtime_;  
    TRTUnique<ICudaEngine> engine_;
    TRTUnique<IExecutionContext> context_;

    // stores binding indices both preset to -1
    int inBinding_  = -1; 
    int outBinding_ = -1;

    // how many bytes the unout and output tensors take 
    size_t inBytes_  = 0;
    size_t outBytes_ = 0;

    // raw pointers for GPU mem
    void* dIn_  = nullptr;
    void* dOut_ = nullptr;

    int outC_ = 0;        // channels per candidate 84 for coco yolos, 56 for pose 
    int outN_ = 0;        // number of candidates 
    bool outIsCN_ = true; // true if output is [1,C,N] ,  false if [1,N,C] - different for diff yolos 

private:
    // readfile helper loads the .plan bytes , just to be used in thisfile
    // returns a vector with raw bytes of the file 
    static std::vector<char> readFile(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Could not open plan file: " + path);

        return std::vector<char>((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
    }
    // helper to figure out which slot index is the input and which is the output
    void findBindings() {
        int nb = engine_->getNbBindings();
        for (int i = 0; i < nb; ++i) {
            if (engine_->bindingIsInput(i)) inBinding_ = i;
            else outBinding_ = i;
        }
        if (inBinding_ < 0 || outBinding_ < 0)
            throw std::runtime_error("Could not find input/output bindings");
    }

    static size_t volume(const Dims& d) {
        // Multiply all dimensions together 
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) {
            int dim = d.d[i];
            v *= static_cast<size_t>((dim > 0) ? dim : 1);
        }
        return v;
    }
    // allocates buff
    void allocateBuffers() {
      // ask tensor rt what the shape needed is
        Dims inDims  = engine_->getBindingDimensions(inBinding_);
        Dims outDims = engine_->getBindingDimensions(outBinding_);
      
        // compute the total number of elements
        size_t inCount  = volume(inDims);
        size_t outCount = volume(outDims);

        // convert element counts into bytes , float32 = 4 bytes 
        inBytes_  = inCount  * sizeof(float);
        outBytes_ = outCount * sizeof(float);

        // allocate buffers using cuda function cudaMalloc
        cudaMalloc(&dIn_,  inBytes_);
        cudaMalloc(&dOut_, outBytes_);
    }

	void inferOutN() {
	    Dims out = engine_->getBindingDimensions(outBinding_);

	    // Common cases:
	    //  - [1, C, N]  (outIsCN_=true)
	    //  - [1, N, C]  (outIsCN_=false)
	    //  - [C, N] or [N, C]
	    if (out.nbDims == 3) {
	        int d1 = out.d[1];
	        int d2 = out.d[2];
	
	        if (d1 <= 0 || d2 <= 0) throw std::runtime_error("Output dims invalid");

	        // Usually N >> C, so bigger dim is N.
	        if (d1 > d2) { // [1, N, C]
	            outIsCN_ = false;
	            outN_ = d1;
	            outC_ = d2;
	        } else {       // [1, C, N]
	            outIsCN_ = true;
	            outC_ = d1;
	            outN_ = d2;
	        }
	        return;
	    }

	    if (out.nbDims == 2) {
	        int d0 = out.d[0];
	        int d1 = out.d[1];
	
	        if (d0 <= 0 || d1 <= 0) throw std::runtime_error("Output dims invalid");
	
	        if (d0 > d1) { // [N, C]
		            outIsCN_ = false;
	            outN_ = d0;
	            outC_ = d1;
	        } else {       // [C, N]
	            outIsCN_ = true;
	            outC_ = d0;
	            outN_ = d1;
	        }
	        return;
	    }

	    throw std::runtime_error("Unexpected output nbDims=" + std::to_string(out.nbDims));
	}


};


// Global engine instance for the whole program
static std::unique_ptr<TRTEngine> gEngine;

extern "C" void trtInit(const std::string& plan) {
    try {
        std::cerr << "[TRT] Loading plan: " << plan << std::endl;
        gEngine.reset(new TRTEngine(plan));
        std::cerr << "[TRT] Engine loaded OK" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[TRT] trtInit failed: " << e.what() << std::endl;
        gEngine.reset();
    }
}

extern "C" cv::Mat trtInfer(const cv::Mat& blob) {
    if (!gEngine) {
        std::cerr << "[TRT] trtInfer called but engine is null\n";
        return cv::Mat();
    }
    return gEngine->infer(blob);
}
