#include "pose.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <exception>
#include <algorithm>
#include <cmath>
#include <cctype>

using Clock=std::chrono::steady_clock;
class VideoSource {
    cv::VideoCapture capture_;
    std::thread worker_;
    std::mutex mutex_;
    cv::Mat latest_;
    std::atomic<bool> running_{true},done_{false};
    std::exception_ptr error_;
public:
    explicit VideoSource(const std::string& source) {
        bool camera=!source.empty()&&source.find_first_not_of("0123456789")==std::string::npos;
        bool opened=camera ? capture_.open(std::stoi(source),cv::CAP_DSHOW) : capture_.open(source);
        if (!opened&&camera) opened=capture_.open(std::stoi(source));
        if (!opened) throw std::runtime_error("Cannot open camera/video: "+source);
        if (camera) { capture_.set(cv::CAP_PROP_FRAME_WIDTH,1280); capture_.set(cv::CAP_PROP_FRAME_HEIGHT,720); capture_.set(cv::CAP_PROP_BUFFERSIZE,1); }
        double fps=capture_.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps)||fps<=0||fps>240) fps=30;
        worker_=std::thread([this,camera,fps] {
            try {
                cv::Mat frame; auto start=Clock::now(); int64_t index=0;
                while (running_&&capture_.read(frame)&&!frame.empty()) {
                    if (!camera) {
                        auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(index++/fps));
                        while (running_&&Clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                    if (!running_) break;
                    std::lock_guard<std::mutex> lock(mutex_); latest_=frame.clone();
                }
            } catch (...) { std::lock_guard<std::mutex> lock(mutex_); error_=std::current_exception(); }
            capture_.release(); done_=true;
        });
    }
    bool take(cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(error_) std::rethrow_exception(error_);
        if(latest_.empty()) return false;
        frame=std::move(latest_); latest_=cv::Mat(); return true;
    }
    bool finished() { std::lock_guard<std::mutex> lock(mutex_); return done_&&latest_.empty(); }
    ~VideoSource() { running_=false; if(worker_.joinable()) worker_.join(); }
};

static std::map<std::string,std::string> parse(int argc,char** argv) {
    std::map<std::string,std::string> options;
    const std::set<std::string> allowed={"source","model","confidence","keypoint","save","headless","frames","help","loop"};
    for(int i=1;i<argc;i++) {
        std::string arg=argv[i];
        if(arg.rfind("--",0)!=0) throw std::runtime_error("Use --name=value: "+arg);
        auto equal=arg.find('='); auto key=arg.substr(2,equal==std::string::npos ? equal : equal-2);
        if(!allowed.count(key)) throw std::runtime_error("Unknown option: "+key);
        bool flag=key=="headless"||key=="help"||key=="loop";
        if(!flag&&(equal==std::string::npos||equal+1==arg.size())) throw std::runtime_error("Missing option value: "+key);
        options[key]=equal==std::string::npos?"true":arg.substr(equal+1);
    }
    return options;
}
static float probability(const std::string& text) {
    float value=std::stof(text);
    if(!std::isfinite(value)||value<=0||value>1) throw std::runtime_error("Confidence must be > 0 and <= 1");
    return value;
}
int main(int argc,char** argv) {
    try {
        auto options=parse(argc,argv);
        auto get=[&](const std::string& key,const std::string& fallback) { return options.count(key)?options.at(key):fallback; };
        if(options.count("help")) {
            std::cout<<"Movira --source=0|video.mp4|photo.jpg --model=models/yolov8n-pose.onnx --confidence=0.35 --keypoint=0.35 --save=result.png --headless --frames=100 --loop\n"; return 0;
        }
        cv::setNumThreads(std::max(1,std::min(4,cv::getNumberOfCPUs())));
        auto source=get("source","0"); auto model=get("model","models/yolov8n-pose.onnx");
        if(!std::filesystem::is_regular_file(model)) throw std::runtime_error("Model missing: "+model+". Set working directory to Movira.");
        bool headless=options.count("headless")>0;
        int limit=std::stoi(get("frames","0")); if(limit<0) throw std::runtime_error("--frames must be >=0");
        float keypoint=probability(get("keypoint","0.35"));
        movira::Estimator estimator(model,probability(get("confidence","0.35")));
        cv::Mat still;
        if(std::filesystem::is_regular_file(source)) {
            std::string ext=std::filesystem::path(source).extension().string();
            std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){ return char(std::tolower(c)); });
            if(ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp"||ext==".webp"||ext==".tif"||ext==".tiff") still=cv::imread(source);
        }
        const std::string title="Movira - Multi-person Pose (Esc: close)";
        if(!headless) { cv::namedWindow(title,cv::WINDOW_NORMAL); cv::resizeWindow(title,1100,720); }
        cv::Mat last; int processed=0; bool closed=false; auto start=Clock::now();
        if(!still.empty()) {
            auto poses=estimator.detect(still); last=still.clone();
            movira::draw(last,poses,keypoint,1/std::chrono::duration<double>(Clock::now()-start).count());
            std::cout<<"Detected people: "<<poses.size()<<std::endl;
        } else {
            do {
                VideoSource video(source);
                while(true) {
                    if(!headless) {
                        int key=cv::waitKey(1);
                        if(key==27||key=='q'||cv::getWindowProperty(title,cv::WND_PROP_VISIBLE)<1) { closed=true; break; }
                    }
                    cv::Mat frame;
                    if(!video.take(frame)) { if(video.finished()) break; std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
                    auto poses=estimator.detect(frame); processed++;
                    double fps=processed/std::chrono::duration<double>(Clock::now()-start).count();
                    last=frame.clone(); movira::draw(last,poses,keypoint,fps);
                    if(!headless) cv::imshow(title,last);
                    if(limit>0&&processed>=limit) break;
                }
            } while(options.count("loop")&&!closed&&(limit==0||processed<limit));
            if(processed==0&&!closed) throw std::runtime_error("No frames decoded");
            std::cout<<"Processed frames: "<<processed<<std::endl;
        }
        if(options.count("save")&&!last.empty()) {
            std::filesystem::path save=options.at("save");
            if(save.has_parent_path()) std::filesystem::create_directories(save.parent_path());
            if(!cv::imwrite(save.string(),last)) throw std::runtime_error("Image save failed");
            std::cout<<"Saved: "<<save.string()<<std::endl;
        }
        if(!headless&&!closed&&!last.empty()) {
            cv::imshow(title,last);
            while(cv::getWindowProperty(title,cv::WND_PROP_VISIBLE)>=1) { int key=cv::waitKey(30); if(key==27||key=='q') break; }
        }
        cv::destroyAllWindows(); return 0;
    } catch(const std::exception& error) { std::cerr<<"Movira: "<<error.what()<<std::endl; return 1; }
}
