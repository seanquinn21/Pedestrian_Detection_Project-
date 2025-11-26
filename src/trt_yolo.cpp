#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#include <vector>
using namespace nvinfer1;

namespace { struct TRTDestroy{ template<typename T> void operator()(T* t)const{ if(t) t->destroy(); } };
template<typename T> using U = std::unique_ptr<T,TRTDestroy>; }

struct Log: ILogger{ void log(Severity s, AsciiChar const* m) noexcept override{ if(s<=Severity::kWARNING) fprintf(stderr,"%s\n",m); } } gLog;

struct TRTEngine{
  U<IRuntime> rt; U<ICudaEngine> eng; U<IExecutionContext> ctx;
  int inB=-1, outB=-1; size_t inBytes=0, outBytes=0; void *dIn=nullptr,*dOut=nullptr;
  int outN=8400;
  TRTEngine(const std::string& plan){
    std::ifstream f(plan, std::ios::binary); std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
    rt.reset(createInferRuntime(gLog)); eng.reset(rt->deserializeCudaEngine(blob.data(), blob.size())); ctx.reset(eng->createExecutionContext());
    for(int i=0;i<eng->getNbBindings();++i) eng->bindingIsInput(i)? inB=i: outB=i;
    auto in=eng->getBindingDimensions(inB), out=eng->getBindingDimensions(outB);
    size_t inCnt=1,outCnt=1; for(int i=0;i<in.nbDims;i++) inCnt*= (in.d[i]>0?in.d[i]:1);
    for(int i=0;i<out.nbDims;i++) outCnt*= (out.d[i]>0?out.d[i]:1);
    inBytes=inCnt*sizeof(float); outBytes=outCnt*sizeof(float);
    if(out.nbDims==3){ if(out.d[1]==84) outN=out.d[2]; else if(out.d[2]==84) outN=out.d[1]; }
    cudaMalloc(&dIn,inBytes); cudaMalloc(&dOut,outBytes);
  }
  ~TRTEngine(){ cudaFree(dIn); cudaFree(dOut); }
};
static std::unique_ptr<TRTEngine> G;
extern "C" void trtInit(const std::string& plan){ G.reset(new TRTEngine(plan)); }
extern "C" cv::Mat trtInfer(const cv::Mat& blob){
  void* b[2]; b[G->inB]=G->dIn; b[G->outB]=G->dOut;
  cudaMemcpy(G->dIn, blob.ptr<float>(), G->inBytes, cudaMemcpyHostToDevice);
  G->ctx->enqueueV2(b, 0, nullptr);
  std::vector<float> h(G->outBytes/sizeof(float)); cudaMemcpy(h.data(), G->dOut, G->outBytes, cudaMemcpyDeviceToHost);
  if(h.size()==(size_t)(84*G->outN)){ cv::Mat m(84, G->outN, CV_32F, h.data()); return m.clone(); }
  cv::Mat raw((int)(h.size()/84),84,CV_32F,h.data()); cv::Mat t; cv::transpose(raw,t); return t.clone();
}
