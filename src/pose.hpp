#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <array>
#include <string>
#include <vector>

namespace movira {
struct Joint { cv::Point2f point; float confidence = 0; };
struct Pose { cv::Rect2f box; float confidence = 0; std::array<Joint,17> joints{}; };
struct Letterbox {
    cv::Size original, resized; int left = 0, top = 0;
    static Letterbox fit(cv::Size original, cv::Size target);
    cv::Point2f restore(float x, float y) const;
};
std::vector<Pose> decode(const cv::Mat& output, const Letterbox& transform, float confidence = .35f, float nms = .45f);
class Estimator {
    cv::dnn::Net net_;
    float confidence_;
public:
    explicit Estimator(const std::string& model, float confidence = .35f);
    std::vector<Pose> detect(const cv::Mat& frame);
};
void draw(cv::Mat& frame, const std::vector<Pose>& poses, float jointThreshold, double fps);
}
