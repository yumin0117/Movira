#include "pose.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
int main(int argc,char** argv) {
    try {
        auto t=movira::Letterbox::fit({1280,720},{640,640});
        require(cv::norm(t.restore(0,140)-cv::Point2f(0,0))<.01,"Letterbox top-left");
        require(cv::norm(t.restore(640,500)-cv::Point2f(1280,720))<.01,"Letterbox bottom-right");
        int dims[]={1,56,4}; cv::Mat output(3,dims,CV_32F,cv::Scalar(0));
        float* data=output.ptr<float>();
        for(int i=0;i<4;i++) {
            data[i]=i==2?450.f:100.f+i; data[4+i]=300; data[8+i]=100; data[12+i]=300;
            data[16+i]=i==3?.1f:.9f-i*.05f;
            for(int k=0;k<17;k++) { data[(5+k*3)*4+i]=data[i]; data[(6+k*3)*4+i]=300; data[(7+k*3)*4+i]=.9f; }
        }
        auto people=movira::decode(output,movira::Letterbox::fit({640,640},{640,640}));
        require(people.size()==2,"Must keep multiple people and suppress duplicate/low confidence");
        require(std::abs(people[1].joints[0].point.x-450)<.01,"Joint channel decoding");
        int tdims[]={1,4,56}; cv::Mat transposed(3,tdims,CV_32F);
        for(int i=0;i<4;i++) for(int c=0;c<56;c++) transposed.ptr<float>()[i*56+c]=data[c*4+i];
        require(movira::decode(transposed,t).size()==2,"Transposed output");
        int invalidDims[]={1,300,57}; cv::Mat invalid(3,invalidDims,CV_32F,cv::Scalar(0));
        bool rejected=false; try { movira::decode(invalid,t); } catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"Reject end-to-end model output");
        cv::Mat canvas(640,640,CV_8UC3,cv::Scalar(0)); movira::Pose hidden;
        for(auto& j:hidden.joints) j={{300,300},.1f}; movira::draw(canvas,{hidden},.35f,30);
        require(canvas.at<cv::Vec3b>(300,300)==cv::Vec3b(0,0,0),"Hide low confidence joints");
        std::cout<<"All decoder, NMS, letterbox and overlay checks passed\n";
        if(argc==3) {
            cv::Mat photo=cv::imread(argv[1]); require(!photo.empty(),"Fixture photo must open");
            cv::resize(photo,photo,{680,384});
            cv::VideoWriter video(argv[2],cv::CAP_OPENCV_MJPEG,cv::VideoWriter::fourcc('M','J','P','G'),30,{680,384});
            require(video.isOpened(),"Fixture video writer must open");
            for(int i=0;i<120;i++) {
                cv::Mat frame; cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,10*std::sin(i/10.),0,1,0);
                cv::warpAffine(photo,frame,transform,photo.size()); video.write(frame);
            }
            std::cout<<"Created moving test video: "<<argv[2]<<'\n';
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
