#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SfM.hpp"
#include <iostream>

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

    // Filter matches
    double min_dist = 100;
    for (const auto& match : matches) {
        if (match.distance < min_dist) min_dist = match.distance;
    }
    std::vector<cv::DMatch> good_matches = SfM::filterMatches(matches, min_dist);

    // Extract points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Estimate Essential Matrix
    cv::Mat essential_matrix = SfM::estimateEssentialMatrix(points1, points2);

    // Recover pose and triangulate
    std::vector<cv::Point3f> points3D;
    SfM::recoverPoseAndTriangulate(essential_matrix, points1, points2, points3D);

    // Output 3D points
    for (const auto& point : points3D) {
        std::cout << "3D Point: " << point << std::endl;
    }

    return 0;
}