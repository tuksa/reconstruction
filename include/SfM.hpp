#ifndef SFM_HPP
#define SFM_HPP

#include <opencv2/opencv.hpp>
#include <vector>


class SfM {
public:
    static std::vector<cv::DMatch> filterMatches(const std::vector<cv::DMatch>& matches, double min_dist);
    static cv::Mat estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);
    static void recoverPoseAndTriangulate(const cv::Mat& essential_matrix,
                                          const std::vector<cv::Point2f>& points1,
                                          const std::vector<cv::Point2f>& points2,
                                          std::vector<cv::Point3f>& points3D);
};

#endif // SFM_HPP