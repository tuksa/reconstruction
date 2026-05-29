#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SfM.hpp"
#include <iostream>
#include <string>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <vector>
#include <map>

#include <thread>
#include <chrono>

// Frame structure to hold per-image data
struct Frame {
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
};

cv::Mat buildCameraMatrix() {
    // Reasonable defaults when no calibration is available:
    // assume fx = fy ≈ max(width, height) and principal point at image centre.
    // double fx = std::max(img.cols, img.rows);
    // double fy = fx;
    // double cx = img.cols / 2.0;
    // double cy = img.rows / 2.0;

    double fx = 2905.88; 
    double fy = 2905.88; 
    double cx = 1416;
    double cy = 1064;


    return (cv::Mat_<double>(3, 3) <<
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1);
}


std::vector<Frame> load_image_sequence(const std::string& directory_path) {
    std::vector<Frame> frames;
    namespace fs = std::filesystem;

    // Check if directory exists to avoid crashes
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Directory does not exist: " << directory_path << std::endl;
        return frames;
    }

    // Read all images in directory
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg"|| entry.path().extension() == ".bmp" || entry.path().extension() == ".JPG") {
            Frame frame;
            frame.image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            // std::cout << "Checking frames" << std::endl;
            
            if (frame.image.empty()) {
                std::cerr << "Failed to load: " << entry.path() << std::endl;
                continue;
            }
            frames.push_back(frame);
        }
    }

    if (frames.size() < 2) {
        std::cerr << "Warning: Less than 2 images found in " << directory_path << std::endl;
    }

    return frames;
}


int main() {
    // Load image sequence
    std::vector<Frame> frames;
    // std::string image_dir = "dinoRing/";
    std::string image_dir = "/home/chiroma/Documents/projects/openMVG_Build/ImageDataset_SceauxCastle/images/";

    // // Read all images in directory
    // for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    //     if (entry.path().extension() == ".png") {
    //         Frame frame;
    //         frame.image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    //         if (frame.image.empty()) {
    //             std::cerr << "Failed to load: " << entry.path() << std::endl;
    //             continue;
    //         }
    //         frames.push_back(frame);
    //     }
    // }

    // if (frames.size() < 2) {
    //     std::cerr << "Not enough images found!" << std::endl;
    //     return -1;
    // }

    cv::Mat K = buildCameraMatrix();
    std::cout << "Camera matrix K:\n" << K << "\n\n";

    frames = load_image_sequence(image_dir);


    // Initialize detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);  // Increased features

    // Detect features in all frames
    for (auto& frame : frames) {
        orb->detectAndCompute(frame.image, cv::noArray(), 
                            frame.keypoints, frame.descriptors);
    }

    // Initialize point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // Match every image against all other images
    for (size_t i = 0; i < frames.size(); ++i) {
        for (size_t j = i + 1; j < frames.size(); ++j) {
            // Match features between frames[i] and frames[j]
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            std::vector<cv::DMatch> matches;
            matcher.match(frames[i].descriptors, frames[j].descriptors, matches);

            // Filter matches
            double min_dist = 50;
            for (const auto& match : matches) {
                if (match.distance < min_dist) min_dist = match.distance;
            }
            std::vector<cv::DMatch> good_matches = SfM::filterMatches(matches, min_dist);

            // Extract matched points
            std::vector<cv::Point2f> points1, points2;
            for (const auto& match : good_matches) {
                points1.push_back(frames[i].keypoints[match.queryIdx].pt);
                points2.push_back(frames[j].keypoints[match.trainIdx].pt);
            }

            // Estimate pose and triangulate
            cv::Mat essential_matrix = SfM::estimateEssentialMatrix(points1, points2);
            std::vector<cv::Point3f> points3D;
            cv::Mat R_rel, t_rel;
            SfM::recoverPoseAndTriangulate(essential_matrix, points1, points2, 
                                          points3D, R_rel, t_rel);

            // Update global pose for frame[j]
            frames[j].R = R_rel * frames[i].R;
            frames[j].t = R_rel * frames[i].t + t_rel;

            // Add points to global cloud
            for (const auto& point : points3D) {
                if (std::isfinite(point.x) && std::isfinite(point.y) && 
                    std::isfinite(point.z) && abs(point.z) < 100.0) {
                    global_cloud->points.emplace_back(point.x, point.y, point.z);
                }
            }

            std::cout << "Processed frames " << i << " and " << j 
                      << ": " << points3D.size() << " points" << std::endl;
        }
    }

    // Update cloud properties
    global_cloud->width = global_cloud->points.size();
    global_cloud->height = 1;
    global_cloud->is_dense = false;

    // Visualize result
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Multi-view Reconstruction"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(global_cloud, "reconstruction");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "reconstruction");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}