#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SfM.hpp"
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thread>
#include <chrono>
#include <algorithm>

// ─────────────────────────────────────────────
// Helper: build camera intrinsic matrix K
// Replace fx, fy, cx, cy with your actual values.
// If you have EXIF data or a calibration file, load from there.
// ─────────────────────────────────────────────
cv::Mat buildCameraMatrix(const cv::Mat& img) {
    // Reasonable defaults when no calibration is available:
    // assume fx = fy ≈ max(width, height) and principal point at image centre.
    double fx = std::max(img.cols, img.rows);
    double fy = fx;
    double cx = img.cols / 2.0;
    double cy = img.rows / 2.0;

    return (cv::Mat_<double>(3, 3) <<
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1);
}

int main() {
    // ── 1. Load images ────────────────────────────────────────────────────────
    const std::string image1_path =
        "/home/chiroma/Documents/projects/openMVG_Build/"
        "ImageDataset_SceauxCastle/images/100_7101.JPG";
    const std::string image2_path =
        "/home/chiroma/Documents/projects/openMVG_Build/"
        "ImageDataset_SceauxCastle/images/100_7102.JPG";

    cv::Mat img1_color = cv::imread(image1_path, cv::IMREAD_COLOR);
    cv::Mat img2_color = cv::imread(image2_path, cv::IMREAD_COLOR);

    if (img1_color.empty() || img2_color.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return -1;
    }

    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1_color, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2_color, img2_gray, cv::COLOR_BGR2GRAY);

    // ── 2. Camera intrinsics ─────────────────────────────────────────────────
    // Both images assumed to share the same camera / intrinsics.
    cv::Mat K = buildCameraMatrix(img1_gray);
    std::cout << "Camera matrix K:\n" << K << "\n\n";

    // ── 3. Detect keypoints & compute descriptors (ORB, 5000 features) ───────
    cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1_gray, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2_gray, cv::noArray(), keypoints2, descriptors2);

    std::cout << "Keypoints – img1: " << keypoints1.size()
              << "  img2: " << keypoints2.size() << std::endl;

    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "Error: No descriptors found!" << std::endl;
        return -1;
    }

    // ── 4. Match with kNN + Lowe's ratio test ────────────────────────────────
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float LOWE_RATIO = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < LOWE_RATIO * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    std::cout << "Good matches after ratio test: " << good_matches.size() << std::endl;

    // Need at least 8 matches for the Essential Matrix (8-point algorithm).
    if (good_matches.size() < 8) {
        std::cerr << "Error: Not enough good matches (" << good_matches.size()
                  << "). Need at least 8." << std::endl;
        return -1;
    }

    // ── 5. Extract matched point coordinates ─────────────────────────────────
    std::vector<cv::Point2f> points1, points2;
    points1.reserve(good_matches.size());
    points2.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // ── 6. Visualise matches ──────────────────────────────────────────────────
    cv::Mat img_matches;
    cv::drawMatches(img1_gray, keypoints1, img2_gray, keypoints2,
                    good_matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matched Keypoints", img_matches);
    cv::waitKey(3000);   // show for 3 s then continue
    cv::destroyAllWindows();

    // ── 7. Estimate Essential Matrix (needs pixel coords + K) ─────────────────
    // cv::findEssentialMat works directly in pixel space when K is provided.
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        points1, points2,
        K,
        cv::RANSAC,
        0.999,   // confidence
        1.0,     // RANSAC pixel threshold
        inlier_mask);

    if (E.empty()) {
        std::cerr << "Error: Essential matrix estimation failed!" << std::endl;
        return -1;
    }

    // Keep only inlier point pairs.
    std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
    for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
            inlier_pts1.push_back(points1[i]);
            inlier_pts2.push_back(points2[i]);
        }
    }
    std::cout << "Inliers after RANSAC: " << inlier_pts1.size() << std::endl;

    // ── 8. Recover camera pose (R, t) from E ─────────────────────────────────
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, inlier_pts1, inlier_pts2, K, R, t);
    std::cout << "recoverPose inliers: " << inliers      << "\n"
              << "Rotation R:\n"         << R             << "\n"
              << "Translation t:\n"      << t             << "\n";

    // ── 9. Triangulate 3-D points ─────────────────────────────────────────────
    // Build projection matrices P1 = K[I|0]  and  P2 = K[R|t]
    cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
    cv::Mat eye3 = cv::Mat::eye(3, 3, CV_64F);
    eye3.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    P1 = K * P1;

    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));
    P2 = K * P2;

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, inlier_pts1, inlier_pts2, points4D);

    // ── 10. Convert homogeneous → Euclidean, filter bad points ───────────────
    std::vector<cv::Point3f> points3D;
    const double Z_MIN    =  0.0;    // discard points behind camera
    const double Z_MAX    = 200.0;   // discard very distant outliers

    for (int i = 0; i < points4D.cols; ++i) {
        float w = points4D.at<float>(3, i);
        if (std::abs(w) < 1e-6f) continue;  // degenerate

        cv::Point3f p(
            points4D.at<float>(0, i) / w,
            points4D.at<float>(1, i) / w,
            points4D.at<float>(2, i) / w);

        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (p.z > Z_MIN && p.z < Z_MAX)
            points3D.push_back(p);
    }

    std::cout << "Valid 3-D points: " << points3D.size() << std::endl;

    // ── 11. Build PCL point cloud ─────────────────────────────────────────────
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& p : points3D) {
        cloud->points.emplace_back(p.x, p.y, p.z);
    }
    cloud->width    = static_cast<uint32_t>(cloud->points.size());
    cloud->height   = 1;
    cloud->is_dense = false;

    if (cloud->points.empty()) {
        std::cerr << "Warning: Point cloud is empty after filtering. "
                     "Check intrinsics or Z thresholds." << std::endl;
        // Don't exit – still show the (empty) viewer so the error is visible.
    }

    std::cout << "Final point cloud size: " << cloud->points.size() << std::endl;

    // ── 12. Visualise with PCLVisualizer ──────────────────────────────────────
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("SfM – 3D Point Cloud"));

    viewer->setBackgroundColor(0.05, 0.05, 0.05);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sfm_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sfm_cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Camera-1 origin (red sphere) and camera-2 centre (blue sphere).
    viewer->addSphere(pcl::PointXYZ(0, 0, 0), 0.05, 1.0, 0.0, 0.0, "cam1");

    pcl::PointXYZ cam2_center(
        static_cast<float>(t.at<double>(0)),
        static_cast<float>(t.at<double>(1)),
        static_cast<float>(t.at<double>(2)));
    viewer->addSphere(cam2_center, 0.05, 0.0, 0.0, 1.0, "cam2");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "SfM process completed successfully." << std::endl;
    return 0;
}