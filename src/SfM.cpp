#include "SfM.hpp"

std::vector<cv::DMatch> SfM::filterMatches(const std::vector<cv::DMatch>& matches, double min_dist) {
    std::vector<cv::DMatch> good_matches;
    for (const auto& match : matches) {
        if (match.distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(match);
        }
    }
    return good_matches;
}

cv::Mat SfM::estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    return cv::findEssentialMat(points1, points2, 1.0, cv::Point2d(0, 0), cv::RANSAC);
}

void SfM::recoverPoseAndTriangulate(const cv::Mat& essential_matrix,
                                    const std::vector<cv::Point2f>& points1,
                                    const std::vector<cv::Point2f>& points2,
                                    std::vector<cv::Point3f>& points3D) {
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, points1, points2, R, t);

    cv::Mat proj1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat proj2(3, 4, CV_64F);
    R.copyTo(proj2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(proj2(cv::Rect(3, 0, 1, 3)));

    cv::Mat points4D;
    cv::triangulatePoints(proj1, proj2, points1, points2, points4D);

    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3);
        points3D.emplace_back(col.at<float>(0), col.at<float>(1), col.at<float>(2));
    }
}