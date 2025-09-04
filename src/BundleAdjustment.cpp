#include "../include/BundleAdjustment.hpp"

void BundleAdjuster::adjust(std::vector<cv::Point3f>& points3D,
                           const std::vector<cv::Point2f>& points1,
                           const std::vector<cv::Point2f>& points2,
                           cv::Mat& R,
                           cv::Mat& t) {
    // Problem setup
    ceres::Problem problem;
    
    // Camera parameters: [angle-axis (3), translation (3), focal length (1)]
    double camera1[7] = {0, 0, 0, 0, 0, 0, 1000}; // First camera at origin
    double camera2[7];

    std::cout << "Initial R: " << R << std::endl;
    std::cout << "Initial t: " << t << std::endl;
    
    // Convert R to angle-axis
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    camera2[0] = rvec.at<double>(0);
    camera2[1] = rvec.at<double>(1);
    camera2[2] = rvec.at<double>(2);
    camera2[3] = t.at<double>(0);
    camera2[4] = t.at<double>(1);
    camera2[5] = t.at<double>(2);
    camera2[6] = 1000; // Initial focal length

    // Add residual blocks for both cameras
    for (size_t i = 0; i < points3D.size(); ++i) {
        double* point = new double[3];
        point[0] = points3D[i].x;
        point[1] = points3D[i].y;
        point[2] = points3D[i].z;

        // Add residual for first camera
        problem.AddResidualBlock(
            ReprojectionError::Create(points1[i].x, points1[i].y),
            nullptr,
            camera1,
            point);

        // Add residual for second camera
        problem.AddResidualBlock(
            ReprojectionError::Create(points2[i].x, points2[i].y),
            nullptr,
            camera2,
            point);
    }

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update camera parameters and 3D points
    cv::Mat rvec2(3, 1, CV_64F);
    rvec2.at<double>(0) = camera2[0];
    rvec2.at<double>(1) = camera2[1];
    rvec2.at<double>(2) = camera2[2];
    cv::Rodrigues(rvec2, R);
    
    t.at<double>(0) = camera2[3];
    t.at<double>(1) = camera2[4];
    t.at<double>(2) = camera2[5];
}