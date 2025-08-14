#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load two images
    cv::Mat img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return -1;
    }

    // Step 1: Detect keypoints and compute descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Step 2: Match descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Filter matches based on distance
    double max_dist = 0, min_dist = 100;
    for (const auto& match : matches) {
        double dist = match.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (const auto& match : matches) {
        if (match.distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(match);
        }
    }

    // Step 3: Estimate Essential Matrix
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, 1.0, cv::Point2d(0, 0), cv::RANSAC);

    // Step 4: Recover pose
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, points1, points2, R, t);

    // Step 5: Triangulate points
    cv::Mat proj1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat proj2(3, 4, CV_64F);
    R.copyTo(proj2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(proj2(cv::Rect(3, 0, 1, 3)));

    cv::Mat points4D;
    cv::triangulatePoints(proj1, proj2, points1, points2, points4D);

    // Convert homogeneous coordinates to 3D
    std::vector<cv::Point3f> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3);
        points3D.emplace_back(col.at<float>(0), col.at<float>(1), col.at<float>(2));
    }

    // Output 3D points
    for (const auto& point : points3D) {
        std::cout << "3D Point: " << point << std::endl;
    }

    return 0;
}