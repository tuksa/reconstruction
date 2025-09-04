#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SfM.hpp"
#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <thread>
#include <chrono>

int main() {
    // Load two images
    std::string image1 = "dinoRing/dinoR0024.png";
    std::string image2 = "dinoRing/dinoR0025.png";
    cv::Mat img1 = cv::imread(image1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(image2, cv::IMREAD_GRAYSCALE);
    // cv::Mat img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);

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

    // Visualize matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the matches
    cv::imshow("Matched Keypoints", img_matches);
    cv::waitKey(0); // Wait for a key press to close the window
    cv::destroyAllWindows();

    // Estimate Essential Matrix
    cv::Mat essential_matrix = SfM::estimateEssentialMatrix(points1, points2);

    cv::Mat R, t;

    // Recover pose and triangulate
    std::vector<cv::Point3f> points3D;
    SfM::recoverPoseAndTriangulate(essential_matrix, points1, points2, points3D, R, t);

    std::cout << "Recovered Rotation:\n" << R << std::endl;
    std::cout << "Recovered Translation:\n" << t << std::endl;
    std::cout << "Number of 3D points: " << points3D.size() << std::endl;

    // // Output 3D points
    // for (const auto& point : points3D) {
    //     std::cout << "3D Point: " << point << std::endl;
    // }

    // // Create a PCL point cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // for (const auto& point : points3D) {
    //     cloud->points.emplace_back(point.x, point.y, point.z);
    // }
    // cloud->width = cloud->points.size();
    // cloud->height = 1; // Unorganized point cloud
    // cloud->is_dense = false;

    // // Visualize the point cloud
    // pcl::visualization::CloudViewer viewer("3D Point Cloud Viewer");
    // viewer.showCloud(cloud);

    // // Keep the viewer open until the user closes it
    // while (!viewer.wasStopped()) {}


    //
    // Create a PCL point cloud with better error handling
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Filter and add valid points only
    for (const auto& point : points3D) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            // Optional: filter out points that are too far
            if (abs(point.z) < 100.0) {  // Adjust threshold as needed
                cloud->points.emplace_back(point.x, point.y, point.z);
            }
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    std::cout << "Final point cloud size: " << cloud->points.size() << std::endl;

    if (cloud->points.empty()) {
        std::cout << "Warning: Point cloud is empty!" << std::endl;
        // return -1;
    }

    // Use PCL Visualizer instead of CloudViewer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Point Cloud"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Keep the viewer running
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // // Test with known points first
    // pcl::PointCloud<pcl::PointXYZ>::Ptr test_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // test_cloud->points.emplace_back(0, 0, 0);
    // test_cloud->points.emplace_back(1, 0, 0);
    // test_cloud->points.emplace_back(0, 1, 0);
    // test_cloud->points.emplace_back(0, 0, 1);
    // test_cloud->width = 4;
    // test_cloud->height = 1;
    // test_cloud->is_dense = false;

    // pcl::visualization::PCLVisualizer::Ptr test_viewer(new pcl::visualization::PCLVisualizer("Test Cloud"));
    // test_viewer->setBackgroundColor(0, 0, 0);
    // test_viewer->addPointCloud<pcl::PointXYZ>(test_cloud, "test");
    // test_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "test");
    // test_viewer->addCoordinateSystem(1.0);

    // while (!test_viewer->wasStopped()) {
    //     test_viewer->spinOnce(100);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }

    std::cout << "SfM process completed successfully." << std::endl;
    return 0;
}