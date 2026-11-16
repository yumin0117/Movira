#include "pose.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace movira {
Letterbox Letterbox::fit(cv::Size original, cv::Size target) {
    double scale = std::min(double(target.width)/original.width, double(target.height)/original.height);
    cv::Size resized(int(std::round(original.width*scale)), int(std::round(original.height*scale)));
    return {original,resized,(target.width-resized.width)/2,(target.height-resized.height)/2};
}
cv::Point2f Letterbox::restore(float x, float y) const {
    return {(x-left)*original.width/resized.width,(y-top)*original.height/resized.height};
}
static float overlap(const cv::Rect2f& a, const cv::Rect2f& b) {
    float intersection = (a&b).area();
    return intersection/std::max(1e-6f,a.area()+b.area()-intersection);
}
std::vector<Pose> decode(const cv::Mat& output, const Letterbox& transform, float confidence, float nms) {
    if (output.dims!=3 || output.size[0]!=1 || output.type()!=CV_32F || !output.isContinuous())
        throw std::runtime_error("Expected contiguous float output [1,56,N] or [1,N,56]");
    bool channelsFirst = output.size[1]==56;
    if (!channelsFirst && output.size[2]!=56)
        throw std::runtime_error("Unsupported model output. Use raw YOLOv8/11 pose, without built-in NMS.");
    int count = channelsFirst ? output.size[2] : output.size[1];
    const float* data = output.ptr<float>();
    auto value = [&](int i,int channel) { return channelsFirst ? data[channel*count+i] : data[i*56+channel]; };
    std::vector<Pose> candidates, result;
    for (int i=0;i<count;i++) {
        float score=value(i,4), x=value(i,0), y=value(i,1), w=value(i,2), h=value(i,3);
        if (!std::isfinite(score) || score<confidence || !std::isfinite(x+y+w+h) || w<=0 || h<=0) continue;
        auto tl=transform.restore(x-w/2,y-h/2), br=transform.restore(x+w/2,y+h/2);
        Pose pose; pose.box=cv::Rect2f(tl,br); pose.confidence=score;
        for (int k=0;k<17;k++) pose.joints[k]={transform.restore(value(i,5+k*3),value(i,6+k*3)),value(i,7+k*3)};
        candidates.push_back(pose);
    }
    std::sort(candidates.begin(),candidates.end(),[](const Pose& a,const Pose& b) { return a.confidence>b.confidence; });
    for (const Pose& pose:candidates) {
        bool duplicate=std::any_of(result.begin(),result.end(),[&](const Pose& kept) { return overlap(pose.box,kept.box)>nms; });
        if (!duplicate) result.push_back(pose);
    }
    return result;
}
Estimator::Estimator(const std::string& model, float confidence): net_(cv::dnn::readNetFromONNX(model)),confidence_(confidence) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}
std::vector<Pose> Estimator::detect(const cv::Mat& frame) {
    // The supplied model has a fixed 640x640 RGB input.
    auto transform=Letterbox::fit(frame.size(),{640,640});
    cv::Mat resized,padded(640,640,CV_8UC3,cv::Scalar(114,114,114));
    cv::resize(frame,resized,transform.resized);
    resized.copyTo(padded(cv::Rect(transform.left,transform.top,resized.cols,resized.rows)));
    net_.setInput(cv::dnn::blobFromImage(padded,1.0/255.0,{640,640},cv::Scalar(),true,false));
    return decode(net_.forward(),transform,confidence_);
}
static bool visible(const Joint& j,float threshold) {
    return j.confidence>=threshold && std::isfinite(j.point.x) && std::isfinite(j.point.y);
}
void draw(cv::Mat& frame,const std::vector<Pose>& poses,float threshold,double fps) {
    static constexpr int bones[][2]={{0,1},{0,2},{1,3},{2,4},{3,5},{4,6},{5,6},{5,7},{7,9},{6,8},{8,10},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}};
    int thickness=std::max(2,frame.cols/400), radius=std::max(3,thickness+1);
    cv::Scalar green(90,255,60), yellow(80,255,230);
    for (const auto& pose:poses) {
        for (const auto& bone:bones) {
            const auto& a=pose.joints[bone[0]]; const auto& b=pose.joints[bone[1]];
            if (visible(a,threshold)&&visible(b,threshold)) cv::line(frame,a.point,b.point,green,thickness,cv::LINE_AA);
        }
        if (visible(pose.joints[5],threshold)&&visible(pose.joints[6],threshold)&&visible(pose.joints[11],threshold)&&visible(pose.joints[12],threshold))
            cv::line(frame,(pose.joints[5].point+pose.joints[6].point)*.5f,(pose.joints[11].point+pose.joints[12].point)*.5f,green,thickness,cv::LINE_AA);
        for (const auto& joint:pose.joints) if (visible(joint,threshold)) {
            cv::circle(frame,joint.point,radius+1,cv::Scalar(20,35,20),-1,cv::LINE_AA);
            cv::circle(frame,joint.point,radius,yellow,-1,cv::LINE_AA);
        }
    }
    char text[100]; std::snprintf(text,sizeof(text),"Movira | People: %zu | FPS: %.1f",poses.size(),fps);
    cv::rectangle(frame,{8,8},{std::min(frame.cols-1,430),43},cv::Scalar(28,28,28),-1);
    cv::putText(frame,text,{16,32},cv::FONT_HERSHEY_SIMPLEX,.6,cv::Scalar(255,255,255),1,cv::LINE_AA);
}
}
